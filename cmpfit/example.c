#include "mpfit.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* This is the private data structure which contains the data points
   and their uncertainties */
struct vars_struct {
  double *x;
  double *y;
  double *ey;
};

/*
 * linear fit function
 *
 * m - number of data points
 * n - number of parameters (2)
 * p - array of fit parameters 
 * dy - array of residuals to be returned
 * vars - private data (struct vars_struct *)
 *
 * RETURNS: error code (0 = success)
 */
int linfunc(int m, int n, double *p, double *dy, double **dvec, void *vars)
{
  int i;
  struct vars_struct *v = (struct vars_struct *) vars;
  double *x, *y, *ey, f;

  x = v->x;
  y = v->y;
  ey = v->ey;

  for (i=0; i<m; i++) {
    f = p[0] - p[1]*x[i];     /* Linear fit function; note f = a - b*x */
    dy[i] = (y[i] - f)/ey[i];
  }

  return 0;
}

/* Test harness routine, which contains test data, invokes mpfit() */
int main(int argc, char *argv[])
{
  /* X - independent variable */
  double x[] = {-1.7237128E+00,1.8712276E+00,-9.6608055E-01,
		-2.8394297E-01,1.3416969E+00,1.3757038E+00,
		-1.3703436E+00,4.2581975E-02,-1.4970151E-01,
		8.2065094E-01};
  /* Y - measured value of dependent quantity */
  double y[] = {1.9000429E-01,6.5807428E+00,1.4582725E+00,
		2.7270851E+00,5.5969253E+00,5.6249280E+00,
		0.787615,3.2599759E+00,2.9771762E+00,
		4.5936475E+00};
  double ey[10];   /* Measurement uncertainty - initialized below */
   
  double p[2] = {1.0, 1.0};           /* Initial conditions */
  double pactual[2] = {3.20, 1.78};   /* Actual values used to make data */
  double perror[2];                   /* Returned parameter errors */      
  int i;
  struct vars_struct v;  /* Private data structure */
  int status;
  mp_result result;

  memset(&result,0,sizeof(result));       /* Zero results structure */
  result.xerror = perror;
  for (i=0; i<10; i++) ey[i] = 0.07;   /* Data errors */

  /* Fill private data structure */
  v.x = x;
  v.y = y;
  v.ey = ey;

  /* Call fitting function for 10 data points and 2 parameters */
  status = mpfit(linfunc, 10, 2, p, 0, 0, (void *) &v, &result);

  printf("*** testlinfit status = %d\n", status);
  /* ... print or use the results of the fitted parametres p[] here! ... */

  return 0;
}


