/* OpenMP + MPI (Hybrid) Version */

#define _XOPEN_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>


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

    fv = 0.0;
	
    for (i=0; i<n-1; i++)   /* rosenbrock */
        fv = fv + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);

    return fv;
}

/* given a point, look for a better one nearby, one coord at a time */
double best_nearby(double delta[MAXVARS], double point[MAXVARS], double prevbest, int nvars,unsigned long * cnt)
{
	double z[MAXVARS];
	double minf, ftmp;
	int i;
	minf = prevbest;
	for (i = 0; i < nvars; i++)
		z[i] = point[i];
	for (i = 0; i < nvars; i++) {
		z[i] = point[i] + delta[i];
        *cnt = *cnt + 1; 
		ftmp = f(z, nvars);
		if (ftmp < minf)
			minf = ftmp;
		else {
			delta[i] = 0.0 - delta[i];
			z[i] = point[i] + delta[i];
            *cnt = *cnt + 1; 
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


int hooke(int nvars, double startpt[MAXVARS], double endpt[MAXVARS], double rho, double epsilon, int itermax,unsigned long * cnt)
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
    *cnt = *cnt + 1; 
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
		newf = best_nearby(delta, newx, fbefore, nvars,cnt);
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
			newf = best_nearby(delta, newx, fbefore, nvars,cnt);
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

//declaration of struct node type
typedef struct data
{
	double best_pt[MAXVARS]; // disp 0
    double best_fx; // disp 2000 
	int best_trial; // disp 2008
	int best_jj; // disp 2012
    unsigned long fevals; //disp 2016

}node;


// creation & declaration of custfun function
void custfun( node *, node *, int *, MPI_Datatype * );

void custfun(node *invec, node *inoutvec, int *len, MPI_Datatype *dtype)
{
	inoutvec->fevals += invec->fevals;
	int i;
	if(invec->best_fx < inoutvec->best_fx)
	{
		for(i = 0; i < *len; i++)
			inoutvec->best_pt[i] = invec->best_pt[i];
		inoutvec->best_fx = invec->best_fx;
		inoutvec->best_jj = invec->best_jj;
		inoutvec->best_trial = invec->best_trial;
	}
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
	
	node data;
	//creation of the custom datatype
	MPI_Datatype customdtype; //the variable in which store the custom datatype
 	MPI_Datatype type[5] = {MPI_DOUBLE, MPI_DOUBLE,MPI_INT,MPI_INT,MPI_UNSIGNED_LONG}; /*array conataining the MPI datatypes to replicate to make each block*/
	int blocklen[5] = {nvars,1,1,1,1}; /* array containing the length of each block */
	MPI_Aint displ[5]; /*array containing the displacement for each block, expressed in bytes. The displacement is the distance 
	between the start of the MPI created and the start of the block*/

	data.best_fx = 1e10;
	data.best_trial = -1;
	data.best_jj = -1;
	data.fevals = 0;
	
	MPI_Op myop;

	
	//initialize comm and declare that it will be threaded
	//instead of using MPI_Init(&argc, &argv)
	int provided;
    MPI_Init_thread(&argc, &argv,MPI_THREAD_SERIALIZED,&provided);
	//check that the library supports the level of parallelism (in terms of threads) we desire
	if (provided < MPI_THREAD_SERIALIZED)
	{
		fprintf(stdout,"Error:MPI threading level is not supported/enough.\n");
		MPI_Finalize();
		return 1;
	}
    

    int size;
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD,&rank); //number of the specific process
    MPI_Comm_size(MPI_COMM_WORLD,&size);  //total number of processes
 

	MPI_Aint baddress;
	MPI_Get_address(&data,&baddress);
	MPI_Get_address(&data.best_pt,&displ[0]);
	MPI_Get_address(&data.best_fx,&displ[1]);
	MPI_Get_address(&data.best_trial,&displ[2]);
	MPI_Get_address(&data.best_jj,&displ[3]);
	MPI_Get_address(&data.fevals,&displ[4]);

	//the distance between the variables of custom datatype
	displ[0] = MPI_Aint_diff(displ[0],baddress); 
	displ[1] = MPI_Aint_diff(displ[1],baddress); 
	displ[2] = MPI_Aint_diff(displ[2],baddress); 
	displ[3] = MPI_Aint_diff(displ[3],baddress); 
	displ[4] = MPI_Aint_diff(displ[4],baddress); 


    //commit the custom datatype
	MPI_Type_create_struct(5,blocklen,displ,type,&customdtype);
	MPI_Type_commit(&customdtype);


   
	for (i = 0; i < MAXVARS; i++) data.best_pt[i] = 0.0;
   

	//ntrials = atoi(argv[1])*1024;	/* number of trials */
    
    //ntrials = 1*1024; /* number of trials */
	ntrials = 65536;	
    //srand48(1);
	
	double const step = ntrials / size; //number of trials per process
    //split the work in chunks
    //so each proccess can evaluate its own part.
    unsigned long start = rank * step; //the number of first trial of process
    unsigned long end = (rank + 1) * step; //the number of last trial of process
    if (rank == size-1) end = ntrials;
  
   // srand48(rank);
	

	MPI_Barrier(MPI_COMM_WORLD); //synchronization
	if (rank == 0) t0 = MPI_Wtime();
	
	#pragma omp parallel private(startpt,endpt,jj,fx) num_threads(2) /*parallel region. Set the number of threads.*/
	{
		unsigned short buf[3];
		double lbest_fx = 1e10; //the local minimum fx of every thread
		double lbest_pt[MAXVARS]; //the endpt of the local minimum fx 
		int lbest_trial = -1; //the trial of the local minimum fx
		int lbest_jj = -1; //the jj of the local minimum fx
		
		unsigned long lfunevals = 0;
        unsigned long * lfunevalsptr = &lfunevals; //stores the number of function evaluations of every thread

	    #pragma omp for schedule(dynamic)
		for (trial = start; trial < end; trial++) 
		{
			buf[0]=0;
			buf[1]=0;
			buf[2]=trial+omp_get_thread_num();

		
			for (i = 0; i < nvars; i++) 
				startpt[i] = 10.0*erand48(buf)/*genRand(&seed)*/-5.0;
		

			jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax,&lfunevals);
        	fx = f(endpt,nvars);


			if (fx < lbest_fx) /*checking if the fx of this trial of the thread is less than the current minimum fx of the thread*/
        	/*if the condition is true, make the appropriate updates*/ 
			{
				lbest_trial = trial;
				lbest_jj = jj;
				lbest_fx = fx;
				for (i = 0; i < nvars; i++)
					lbest_pt[i] = endpt[i];
			}//end of if
		}// end of worksharing

		#pragma omp critical//no two threads will simultaneously be in the critical section
		{
		    data.fevals += *lfunevalsptr; /*adding the number of function evaluations of this thread to the shared variable of total function evaluations of the process*/
			if (lbest_fx < data.best_fx)  /*checking if the best fx of this thread is less than the current minimum fx of the process*/
			/*if the condition is true, make the appropriate updates*/
			{
				data.best_trial = lbest_trial;
				data.best_jj = lbest_jj;
				data.best_fx = lbest_fx;
				for (i = 0; i < nvars; i++)
					data.best_pt[i] = lbest_pt[i];
			} //end of if
        } //end of critical region
	} // end of parallel region

	
	
	node output; /*this struct will gonna store the whole information about the total number of function evaluations and the compution of minimum fx*/
	for(i = 0; i < nvars; i++)
		output.best_pt[i] = 0.0;
	
	output.best_fx = 1;
	MPI_Op_create((MPI_User_function *)custfun,0,&myop); //creation of user-defined operation myop
	MPI_Reduce(&data,&output,1,customdtype,myop,0,MPI_COMM_WORLD); 
	MPI_Barrier(MPI_COMM_WORLD); 
	if (rank == 0) t1 = MPI_Wtime();
	MPI_Op_free( &myop ); //dealloc the memory space of the user-defined operation myop

    if(rank == 0) /*only the root process prints the variables of (struct) node output*/
    {
		printf("\n\nFINAL RESULTS:\n");
		printf("Elapsed time = %.3lf s\n", t1-t0);
		printf("Total number of trials = %d\n", ntrials);
		printf("Total number of function evaluations = %ld\n", output.fevals);
		printf("Best result at trial %d used %d iterations, and returned\n",output.best_trial,output.best_jj);
		for (i = 0; i < nvars; i++) {
			printf("x[%3d] = %15.7le \n", i, output.best_pt[i]);
		}
		printf("f(x) = %15.7le\n", output.best_fx);

    }
	
	MPI_Type_free(&customdtype); //dealloc the memory space of the custom dataype
	MPI_Finalize(); //clean up at the end


	return 0;
}
