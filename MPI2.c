/* MPI --> 2nd Version --> MPI_Gather and MPI_Bcast*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>


#define MAXVARS		(250)	/* max # of variables	     */
#define RHO_BEGIN	(0.9)	/* stepsize geometric shrink */
#define EPSMIN		(1E-6)	/* ending value of stepsize  */
#define IMAX		(5000)	/* max # of iterations	     */

/* global variables */
unsigned long funevals = 0;


/* Rosenbrock classic parabolic valley ("banana") function */
double f(double *x, int n)
{
    double fv;
    int i;

    funevals++;
    fv = 0.0;
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

    return fv;
}

/* given a point, look for a better one nearby, one coord at a time */
double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars)
{
	double z[MAXVARS];
	double minf, ftmp;
	int i;
	minf = prevbest;
	for (i = 0; i < nvars; i++)
		z[i] = point[i];
	for (i = 0; i < nvars; i++) {
		z[i] = point[i] + delta[i];
		ftmp = f(z, nvars);
		if (ftmp < minf)
			minf = ftmp;
		else {
			delta[i] = 0.0 - delta[i];
			z[i] = point[i] + delta[i];
			ftmp = f(z, nvars);
			if (ftmp < minf)
				minf = ftmp;
			else
				z[i] = point[i];
		}
	}
	for (i = 0; i < nvars; i++)
		point[i] = z[i];

	return (minf);
}


int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax)
{
	double delta[MAXVARS];
	double newf, fbefore, steplength, tmp;
	double xbefore[MAXVARS], newx[MAXVARS];
	int i, j, keep;
	int iters, iadj;

	for (i = 0; i < nvars; i++) {
		newx[i] = xbefore[i] = startpt[i];
		delta[i] = fabs(startpt[i] * rho);
		if (delta[i] == 0.0)
			delta[i] = rho;
	}
	iadj = 0;
	steplength = rho;
	iters = 0;
	fbefore = f(newx, nvars);
	newf = fbefore;
	while ((iters < itermax) && (steplength > epsilon)) {
		iters++;
		iadj++;
#if DEBUG
		printf("\nAfter %5d funevals, f(x) =  %.4le at\n", funevals, fbefore);
		for (j = 0; j < nvars; j++)
			printf("   x[%2d] = %.4le\n", j, xbefore[j]);
#endif
		/* find best new point, one coord at a time */
		for (i = 0; i < nvars; i++) {
			newx[i] = xbefore[i];
		}
		newf = best_nearby(delta, newx, fbefore, nvars);
		/* if we made some improvements, pursue that direction */
		keep = 1;
		while ((newf < fbefore) && (keep == 1)) {
			iadj = 0;
			for (i = 0; i < nvars; i++) {
				/* firstly, arrange the sign of delta[] */
				if (newx[i] <= xbefore[i])
					delta[i] = 0.0 - fabs(delta[i]);
				else
					delta[i] = fabs(delta[i]);
				/* now, move further in this direction */
				tmp = xbefore[i];
				xbefore[i] = newx[i];
				newx[i] = newx[i] + newx[i] - tmp;
			}
			fbefore = newf;
			newf = best_nearby(delta, newx, fbefore, nvars);
			/* if the further (optimistic) move was bad.... */
			if (newf >= fbefore)
				break;

			/* make sure that the differences between the new */
			/* and the old points are due to actual */
			/* displacements; beware of roundoff errors that */
			/* might cause newf < fbefore */
			keep = 0;
			for (i = 0; i < nvars; i++) {
				keep = 1;
				if (fabs(newx[i] - xbefore[i]) > (0.5 * fabs(delta[i])))
					break;
				else
					keep = 0;
			}
		}
		if ((steplength >= epsilon) && (newf >= fbefore)) {
			steplength = steplength * rho;
			for (i = 0; i < nvars; i++) {
				delta[i] *= rho;
			}
		}
	}
	for (i = 0; i < nvars; i++)
		endpt[i] = xbefore[i];

	return (iters);
}


double get_wtime(void)
{
    struct timeval t;

    gettimeofday(&t, NULL);

    return (double)t.tv_sec + (double)t.tv_usec*1.0e-6;
}






int main(int argc, char *argv[])
{
	double startpt[MAXVARS], endpt[MAXVARS];
	int itermax = IMAX;
	double rho = RHO_BEGIN;
	double epsilon = EPSMIN;
	int nvars;
	int trial, ntrials;
	double fx;
	int i, jj;
	double t0, t1;

	nvars = 32;		/* number of variables (problem dimension) */
	
	double best_fx = 1e10;
	double best_pt[MAXVARS];
	int best_trial = -1;
	int best_jj = -1;

	
	//initialize comm
    MPI_Init(&argc, &argv); //initialize the environment
    

    int size;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank); //number of the specific process
    MPI_Comm_size(MPI_COMM_WORLD,&size); //total number of processes

	for (i = 0; i < MAXVARS; i++) best_pt[i] = 0.0;
   

	//ntrials = 1*1024; /* number of trials */
	ntrials = 65536;	
	//srand48(1);

    double const step = ntrials / size; //number of trials per process
    //split the work in chunks
    //so each proccess can evaluate its own part.
    unsigned long start = rank * step; //the number of first trial of process
    unsigned long end = (rank + 1) * step; //the number of last trial of process
    if (rank == size-1) end = ntrials;
  
    //MTRand r = seedRand(rank);

    srand48(rank);
	MPI_Barrier(MPI_COMM_WORLD); //synchronization
	t0 = MPI_Wtime();
	//printf("starting to calculate from %d\n",rank);
	for (trial = start; trial < end; trial++)
	{
		
		for (i = 0; i < nvars; i++) 
	          startpt[i] = 10.0*drand48()-5.0;
		//printf("%d)here\n",trial);
		jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax);
		//printf("%d)here2\n",trial);
        fx = f(endpt,nvars);


		if (fx < best_fx) /*checking if the fx of this trial of the process is less than the current minimum fx of the process*/
		/*if the condition is true, make the appropriate updates in the struct that stores the information of the process*/
		{
			best_trial = trial;
			best_jj = jj;
			best_fx = fx;
			for (i = 0; i < nvars; i++)
				best_pt[i] = endpt[i];
		}
	}
    double fxarr[size]; /*at this array, the root process will gonna store the minimum fx of every process*/
    double fxbuff = 0.0;
    long fsum = 0; /*at this variable, the root process will gonna store the total number of function evaluations*/

    MPI_Gather(&best_fx,1,MPI_DOUBLE,fxarr,1,MPI_DOUBLE,0,MPI_COMM_WORLD);/*the root process receives the variables best_fx from all processes and stores them at the array fxbuff */
    MPI_Reduce(&funevals,&fsum,1,MPI_LONG,MPI_SUM,0,MPI_COMM_WORLD);/*the root process receives the variables funevals from all processes and store the sum of them at the variable fsum*/
    
	if(rank==0) /*the root process traverses the fxarr array and stores the minimum value of fx of all processes at the variable fxbuff*/
    {
        fxbuff = best_fx;
        for(int i = 0; i<size; i++)
        {
            if(fxarr[i]<fxbuff)
               fxbuff = fxarr[i];
           
        }
    }
    
    MPI_Bcast(&fxbuff,1,MPI_DOUBLE,0,MPI_COMM_WORLD); /*the root process sends the fxbuff to all the other processes */
    MPI_Bcast(&fsum,1,MPI_LONG,0,MPI_COMM_WORLD); /*the root process sends the fsum to all the other processes*/
	MPI_Barrier(MPI_COMM_WORLD); //synchronization
	//printf("ending to calculate from %d\n",rank);
	t1 = MPI_Wtime();
    
    if(best_fx == fxbuff) //every process check if the minimum fx is its fx value
	/*if the condition is true, then the process prints the minimum value, the information about the calculation and the total number of function evaluations of all processes*/
    {
    printf("\n\nFINAL RESULTS:\n");
	printf("Elapsed time = %.3lf s\n", t1-t0);
	printf("Total number of trials = %d\n", ntrials);
	printf("Total number of function evaluations = %ld\n", fsum);
	printf("Best result at trial %d used %d iterations, and returned\n", best_trial,best_jj);
	for (i = 0; i < nvars; i++) {
		printf("x[%3d] = %15.7le \n", i, best_pt[i]);
	}
	printf("f(x) = %15.7le\n", best_fx);

    }

	MPI_Finalize();  //clean up at the end

	return 0;
}