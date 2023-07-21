/* OpenMP Version */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

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

	double best_fx = 1e10;
	double best_pt[MAXVARS];
	int best_trial = -1;
	int best_jj = -1;

	for (i = 0; i < MAXVARS; i++) best_pt[i] = 0.0;

	//ntrials = 1*1024; /* number of trials */
	ntrials = 65536;	
	nvars = 32;		/* number of variables (problem dimension) */
	t0 = get_wtime();
    
	#pragma omp parallel private(jj,fx,startpt,endpt,i) num_threads(2) /*parallel region. Set the number of threads.*/
	{
		unsigned short generatorbuf[3];
		generatorbuf[0] = 0;
		generatorbuf[1] = 0;
		
        unsigned long lfunevals = 0;
        unsigned long * lfunevalsptr = &lfunevals; //the number of function evaluations of every thread
        double lbest_fx = 1e10; //the local minimum fx of every thread
		double lbest_pt[MAXVARS]; //the endpt of the local minimum fx 
		int lbest_trial = -1; //the trial of the local minimum fx
		int lbest_jj = -1; //the jj of the local minimum fx 


		#pragma omp for //worksharing
		for ( trial = 0; trial < ntrials; trial++) 
		{
             generatorbuf[2] = trial + omp_get_thread_num();
		     for (i = 0; i < nvars; i++) 
			     startpt[i] = 10.0*erand48(generatorbuf)-5.0;
		    jj = hooke(nvars, startpt, endpt, rho, epsilon, itermax,lfunevalsptr);
		    fx = f(endpt, nvars);
            lfunevals++;
		
			if (fx < lbest_fx) /*checking if the fx of this trial of the thread is less than the current minimum fx of the thread*/
            /*if the condition is true, make the appropriate updates*/
			{
			   lbest_trial = trial;
			   lbest_jj = jj;
			   lbest_fx = fx;
			   for (i = 0; i < nvars; i++)
				   lbest_pt[i] = endpt[i];
		    } 
		
		
	    } // end of worksharing


	    #pragma omp critical //no two threads will simultaneously be in the critical section
		{
            funevals += *lfunevalsptr; /*adding the number of function evaluations of this thread to the shared variable of total function evaluations*/
			if (lbest_fx < best_fx) /*checking if the best fx of this thread is less than the current minimum fx*/
			//if the condition is true, make the appropriate updates
			{
			    best_trial = lbest_trial;
			    best_jj = lbest_jj;
			    best_fx = lbest_fx;
			    for (i = 0; i < nvars; i++)
				    best_pt[i] = lbest_pt[i];
		    }//end of if
		}//end of critical region
    } // end of parallel region
	
	t1 = get_wtime();
	//only the master thread continues

	printf("\n\nFINAL RESULTS:\n");
	printf("Elapsed time = %.3lf s\n", t1-t0);
	printf("Total number of trials = %d\n", ntrials);
	printf("Total number of function evaluations = %ld\n", funevals);
	printf("Best result at trial %d used %d iterations, and returned\n", best_trial, best_jj);
	for (i = 0; i < nvars; i++) 
		printf("x[%3d] = %15.7le \n", i, best_pt[i]);
	
	printf("f(x) = %15.7le\n", best_fx);

	return 0;
}


