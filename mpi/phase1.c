#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>
#include <mkl.h>
#include <math.h>
#include <string.h>
#include "qr2.h"
#include "utility.h"
void mat2csv(int n, int k, double *A, int ldA, char *filename);
// X is d * n (distributed) array
// Y is n (replicated) array
// A is n * k (distributed) array
// A is overwritten by K*A
void kernel_rbf(int rank, int m, int n, int d, float *Xbuf, int ldXbuf, float *X, int ldX, float *Ybuf, float *Y, double gamma, float *K, int ldK)
{
    int i,j,l;
    float *temp = (float*) malloc(sizeof(float) * (m+1) * (n+1));
    cblas_sgemm (CblasColMajor, CblasTrans, CblasNoTrans, m, n, d, 2.0, Xbuf, ldXbuf, X, ldX, 0.0, temp, m);
    float *XISQR = (float*) malloc(sizeof(float) * m );
    float *XJSQR = (float*) malloc(sizeof(float) * n );
    for (i = 0; i < m; i++ ) {
	float acc = 0;
	for (l = 0; l < d; l++) {
	    acc += Xbuf[l + i*ldXbuf] * Xbuf[l + i*ldXbuf];
	}
	XISQR[i] = acc;
    }
    for (j = 0; j < n; j++ ) {
	float acc = 0;
	for (l = 0; l < d; l++) {
	    acc += X[l + j*ldX] * X[l + j*ldX];
	}
	XJSQR[j] = acc;
    }
    for(i = 0; i < m; ++i)
    {
        for(j = 0; j < n; ++j)
        {
            K[i + ldK * j] = exp(-gamma * (XISQR[i] - temp[i+j*m] + XJSQR[j])) * Ybuf[i] * Y[j];
        }
    }
    free(XISQR);
    free(XJSQR);
    free(temp);
    
}

// X is d * n (distributed) array
// // Y is n (replicated) array
// // A is n * k (distributed) array
// // A is overwritten by K*A
void kernel_matmul(int np, int rank, long long int n, int d, int k, float *X, int ldX, float *Y, double gamma, float *A, int ldA)
{
    long long int cbegin = ((n * rank)/np);
    long long int cend = (n * (rank + 1)/np);
    int csize = cend - cbegin;
    int i;
    double start, end;
    float *Abuf = (float*)malloc(sizeof(float) * csize * k);
    for(i = 0; i < np; ++i)
    {
        long long int rbegin = ((n * i)/np);
        long long int rend = (n * (i + 1)/np);
        int rsize = rend - rbegin;
        start = MPI_Wtime();
        float *Xbuf = (float*) malloc (sizeof(float) * rsize * d);
        float *Ybuf = (float*) malloc (sizeof(float) * rsize);
        float *W = (float*)malloc(sizeof(float) * rsize * csize);
        if(rank == i)
        {
            LAPACKE_slacpy(LAPACK_COL_MAJOR, 'P', d, rsize, X, d, Xbuf, d);
            cblas_scopy (rsize, Y, 1, Ybuf, 1);
        }
        MPI_Bcast(Xbuf, (rsize * d), MPI_FLOAT, i, MPI_COMM_WORLD);
        MPI_Bcast(Ybuf, rsize, MPI_FLOAT, i, MPI_COMM_WORLD);
        kernel_rbf(rank, rsize, csize, d, Xbuf, d, X, ldX, Ybuf, Y, gamma, W, rsize);
        end = MPI_Wtime();
        if(rank == 0)
            printf("rank %d :: kernel_matmul :: The time taken to generate the Kernel is %f\n",rank, end - start);
        start = MPI_Wtime();
        float *kai = (float*)malloc(sizeof(float) * rsize * k);
        float *ka = (float*)malloc(sizeof(float) * rsize * k);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rsize, k, csize, 1.0, W, rsize, A, ldA, 0.0, kai, rsize);
        MPI_Reduce(kai, ka, (rsize * k), MPI_FLOAT, MPI_SUM, i, MPI_COMM_WORLD);
        if(rank == i)
            LAPACKE_slacpy(LAPACK_COL_MAJOR, 'P', csize, k, ka, rsize, Abuf, rsize);
        end = MPI_Wtime();
        if(rank == 0)
            printf("rank %d :: kernel_matmul :: The time taken for K*A is : %f\n",rank, end - start);
        free(Xbuf);
        free(kai);
        free(ka);
        free(W);
    }
    if (ldA!=csize) {
        printf("ERROOROROROROORORORO\n!");
        assert(0);
    }
    LAPACKE_slacpy(LAPACK_COL_MAJOR, 'P', csize, k, Abuf, csize, A, ldA);
    free(Abuf);
}

// A is n * k (distributed)
// A should be overwriten by Q
//qr function is already defined in the header file qr2.h
/*void qr(int n, int k, float *A, int ldA);*/

// A, B are n*k (distributed)
// C = A' * B (replicated)
// C is k*k
void inner_product(int rank, int ln, int k, float *A, int ldA, float *B, int ldB, float *C, int ldC)
{
    double start, end;
    start = MPI_Wtime();
    float *Ci = (float*) malloc( sizeof(float) * k * k );
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans, k, k, ln, 1.0, A, ldA, B, ldB, 0.0, Ci, ldC);
    MPI_Reduce(Ci, C, (k * k), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(C, (k * k), MPI_FLOAT, 0, MPI_COMM_WORLD);
    free(Ci);
    end = MPI_Wtime();
    if(rank == 0)
        printf("rank %d :: inner_product :: The time taken for inner product is : %f\n",rank, end - start);
}

// A, B are n*k (distributed) array
// X is d * n (distributed) array
// fnorm computes the norm(K-A*B')
// fnorm is float scalar (replicated)
float fnorm(int np, int rank, long long int n, int d, int k, float *A, int ldA, float *B, int ldB, float *X, int ldX, float *Y, double gamma, float *normK)
{
    long long int cbegin = ((n * rank)/np);
    long long int cend = (n * (rank + 1)/np);
    int csize = cend - cbegin;
    int i, j, l;
    double start, end;
    float fnorm_i = 0.0;
    float fnorm = 0.0;
    float nn_i = 0.0;
    float nn = 0.0;
    
    for(i = 0; i < np; ++i)
    {
        long long int rbegin = ((n * i)/np);
        long long int rend = (n * (i + 1)/np);
        int rsize = rend - rbegin;
        float *Xbuf = (float*) malloc (sizeof(float) * rsize * d);
        float *Ybuf = (float*) malloc (sizeof(float) * rsize);
        float *W = (float*)malloc(sizeof(float) * rsize * csize);
        float *Abuf = (float*) malloc (sizeof(float) *rsize * k);
        if(rank == i)
        {
            LAPACKE_slacpy(LAPACK_COL_MAJOR, 'P', d, rsize, X, d, Xbuf, d);
            cblas_scopy (rsize, Y, 1, Ybuf, 1);
        }
        MPI_Bcast(Xbuf, (rsize * d), MPI_FLOAT, i, MPI_COMM_WORLD);
        MPI_Bcast(Ybuf, rsize, MPI_FLOAT, i, MPI_COMM_WORLD);
        kernel_rbf(rank, rsize, csize, d, Xbuf, d, X, ldX, Ybuf, Y, gamma, W, rsize);
        for(j=0; j<rsize; ++j){
            for(l=0; l<csize; ++l){
                float result = (W[j + l * rsize] * W[j + l * rsize]);
                nn_i = nn_i + result;
            }
        }
        if (rank == i)
        {
            LAPACKE_slacpy(LAPACK_COL_MAJOR, 'P', rsize, k, A, ldA, Abuf, rsize);
        }
        MPI_Bcast(Abuf, rsize*k, MPI_FLOAT, i, MPI_COMM_WORLD);
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans, rsize, csize, k, 1.0, Abuf, rsize, B, ldB, -1.0, W, rsize);
        for(j=0; j<rsize; ++j)
        {
            for(l=0; l<csize; ++l)
            {
                float result = (W[j + l * rsize] * W[j + l * rsize]);
                fnorm_i = fnorm_i + result;
            }
        }
        free(W);
        free(Xbuf);
        free(Ybuf);
        free(Abuf);
    }
    MPI_Reduce(&fnorm_i, &fnorm, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nn_i, normK, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank==0) *normK = sqrt(*normK);
    return sqrt(fnorm);
}

float gaussrand()
{
    static float V1, V2, S;
    static int phase = 0;
    float X;
    
    if(phase == 0) {
        do {
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    return X;
}

void mat2csv(int n, int k, double *A, int ldA, char *filename)
{
    int i, j;
    FILE *fp = fopen(filename, "w+");
    for(i=0; i<n; ++i)
    {
        for(j=0; j<k; ++j)
        {
            fprintf(fp, "%.16le", A[i + j * ldA]);
            if (j!=k-1) fprintf(fp, ",");
            else fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

int readfileY(char* path,double* mat)
{
    
    FILE *fp=fopen(path,"r");
    if(fp == NULL)
    {
        return 1;
    }
    char temp[1024];
    int j=0;
    while(!feof(fp))
    {
        
        fgets(temp,1024,fp);
        temp[strlen(temp)-1]= '\0';
        
        mat[j] = atof(temp);
        j++;
    }
    
    return 1;
}



void writeModel(int k, double *A, char *filename,double b,double gamma,double *a,double *Y,int *AS,int iAS)
{
    int i, j;
    FILE *fp = fopen(filename, "w+");
    fprintf(fp,"%.16lf,%.16lf\n",b,gamma);
    for(i=0; i<iAS; ++i)
    {
        fprintf(fp,"%.16lf,",a[AS[i]]*Y[AS[i]]);
        for(j=0; j<k; ++j)
        {
            fprintf(fp, "%.16lf", A[j+AS[i]*k]);
            if (j!=k-1) fprintf(fp, ",");
            else fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

void matTrans(double* mat, int m,int n,double* transMat)
{
    int i,j;
    
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            transMat[j+i*m]=mat[i+j*m];
        }
    }
    
}

void matAdd(double *mat1,double *mat2,int m,int n)
{
    int i,j;
    for(i=0;i<m;i++)
    {
        for(j=0;j<n;j++)
        {
            mat1[j+i*m]=(mat1[j+i*m]+mat2[j+i*m])/2.0;
        }
    }
}

void poiVecSub(double poi,double *vec,int n,double *result)
{
    for(int i=0;i<n;i++)
    {
        result[i] = poi-vec[i];
    }
}

void setUTri2Zero(double* mat, int d)
{
    for(int i=0;i<d;i++)
    {
        for(int j=0;j<d;j++)
        {
            if(i<j)
            {
                mat[j+i*d]=0;
            }
        }
    }
}

void fill(double *mat, double value, long long int n)
{
    for(int i=0;i<n;i++)
    {
        mat[i] = value;
    }
}

double vecvec_dot_mpi(double *a,double *b,int n)
{
    double c = cblas_ddot(n,a,1,b,1);
    double gc = 0;
    MPI_Allreduce(&c, &gc, 1, MPI_DOUBLE, MPI_SUM,MPI_COMM_WORLD);
    return gc;
}

void matvec_dot_mpi(double *a,double *b,int m,int n,double *d)
{
    double c[n];
    cblas_dgemv(CblasRowMajor,CblasTrans,m,n,1.0,a,n,b,1,0,c,1);
    LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, 1, c, 1, d, 1);
    MPI_Allreduce(c,d,n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

void vecVecSub(double *vec1,double *vec2,int n,double *result)
{
    for(int i=0;i<n;i++)
    {
        result[i] = vec1[i]-vec2[i];
    }
}

double vecSum(double *a,int n)
{
    double sum=0;
    for(int i=0;i<n;i++)
    {
        sum+=a[i];
    }
    return sum;
}

double norm_mpi(double* a,int m,int n)
{
    double sum=0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            sum+=a[i*n+j]*a[i*n+j];
        }
    }
    double c=0.0;
    MPI_Allreduce(&sum,&c,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    return sqrt(c);
}


double norm(double* a,int m,int n)
{
    double sum = 0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            sum+=a[i*n+j]*a[i*n+j];
        }
    }
    return sqrt(sum);
}

double normsquare(double* a,int m,int n)
{
    double sum = 0;
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            sum+=a[i*n+j]*a[i*n+j];
        }
    }
    return sum;
}


void diagScalarMat(double *D,double *Z,int m,int n,double *result)
{
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            result[i*n+j]=(1.0/D[i])*Z[i*n+j];
        }
    }
}

void SMWSolve(double *Z,double *D,double *N,double *b,int nn,int d)
{
    int rank,procnum;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&procnum);
    
    double c[nn];
    for(int i=0;i<nn;i++)
    {
        b[i]=b[i]/D[i];
        c[i] = b[i];
    }
    
    double bb[d];
    matvec_dot_mpi(Z,b,nn,d,bb);
    
    if(rank==0)
    {
        
        LAPACKE_dpotrs(LAPACK_ROW_MAJOR,'U',d,1,N,d,bb,1);
        
    }
    
    MPI_Bcast(bb,d,MPI_DOUBLE,0,MPI_COMM_WORLD);
    double zbbtmp[nn];
    cblas_dgemv(CblasRowMajor,CblasNoTrans,nn,d,1.0,Z,d,bb,1,0,zbbtmp,1);
    for(int i=0;i<nn;i++)
    {
        b[i] = zbbtmp[i]/D[i];
        b[i] = c[i]-b[i];
    }
    
}

void NewtonStep(double *Z,double *D, double *N, double C,double *a,double *X,double *S,double *Xi,double *r1,double *r2,double *r3,double *r4,int nn,int d)
{
    double r5[nn];
    double r7[nn];
    for(int i=0;i<nn;i++)
    {
        r5[i] = r1[i]-r3[i]/X[i]+r4[i]/(C-X[i]);
        r7[i] = r5[i];
    }
    SMWSolve(Z,D,N,r7,nn,d);
    double r6[1];
    double ar7tmp = vecvec_dot_mpi(a,r7,nn);
    double b[nn];
    r6[0] = r2[0]+ar7tmp;
    for(int i=0;i<nn;i++)
    {
        
        b[i] = a[i];
    }
    
    SMWSolve(Z,D,N,b,nn,d);
    double abtmp=vecvec_dot_mpi(a,b,nn);
    r2[0]=r6[0]/abtmp;
    for(int i=0;i<nn;i++)
    {
        r1[i] = a[i]*r2[0]-r5[i];
    }
    SMWSolve(Z,D,N,r1,nn,d);
    for(int i=0;i<nn;i++)
    {
        r3[i] = (r3[i]-S[i]*r1[i])/X[i];
        r4[i] = (r4[i]+Xi[i]*r1[i])/(C-X[i]);
    }
}

double min(double a,double b)
{
    return a<b?a:b;
}

void MPC(double *Z,double *a,double C,double gamma, int b, int n,int d,double* X, double* Xi)
{
    int rank,procnum;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&procnum);
    if(rank==0)
    {
        printf("NumProcess:%d\n",procnum);
    }
    int gn = 0;
    MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
    printf("MPC:R[%d]:gn=%d n=%d d=%d\n",rank,gn,n,d);
    //double X[n];
    fill(X,C/10.0,n);
    double y = 0.0;
    double S[n];
    fill(S,1.0,n);
    //double Xi[n];
    fill(Xi,1.0,n);
    double e[n];
    fill(e,1.0,n);
    double *q=(double*)malloc(sizeof(double)*n);
    fill(q,0.0,n);
    double *qtmp = (double*)malloc(sizeof(double)*d);
    matvec_dot_mpi(Z,e,n,d,qtmp);
    cblas_dgemv(CblasRowMajor,CblasNoTrans,n,d,1.0,Z,d,qtmp,1,0,q,1);
    double qq,qe;
    qq=vecvec_dot_mpi(q,q,n);
    qe=vecvec_dot_mpi(q,e,n);
    double ox=qe/qq;
    //printf("qe/qq=%lf\n",ox);
    if(ox<0.99*C&&ox>0.01*C)
    {
        if(rank==0)
        {
            printf("Initializing X=ox=%lf\n",ox);
        }
        fill(X,ox,n);
    }
    else if(ox>0.99*C)
    {
        if(rank==0)
        {
            printf("Initializing X=0.99C\n");
        }
        fill(X,0.99*C,n);
    }
    else if(ox<0.01*C)
    {
        if(rank==0)
        {
            printf("Initializing X=0.01*C\n");
        }
        fill(X,0.01*C,n);
    }
    double r1[n];
    fill(r1,0.0,n);
    double r2[1];
    r2[0]=0.0;
    double r3[n];
    fill(r3,0.0,n);
    double r4[n];
    fill(r4,0.0,n);
    double raff1[n];
    fill(raff1,0.0,n);
    double raff2[1];
    fill(raff2,0.0,1);
    double raff3[n];
    fill(raff3,0.0,n);
    double raff4[n];
    fill(raff4,0.0,n);
    
    int iter=1;
    while(1)
    {
        double mu;
        double cxtmp[n];
        poiVecSub(C,X,n,cxtmp);
        mu = (vecvec_dot_mpi(X,S,n)+vecvec_dot_mpi(cxtmp,Xi,n))/(2*gn);
        double *dx = (double*)malloc(sizeof(double)*n);
        dx=r1;
        double *dy = (double*)malloc(sizeof(double)*1);
        dy=r2;
        double *ds = (double*)malloc(sizeof(double)*n);
        ds=r3;
        double *dxi = (double*)malloc(sizeof(double)*n);
        dxi=r4;
        
        //==== predictor ====
        double zxtmp[d];
        matvec_dot_mpi(Z,X,n,d,zxtmp);
        double zzxtmp[n];
        cblas_dgemv(CblasRowMajor,CblasNoTrans,n,d,1.0,Z,d,zxtmp,1,0,zzxtmp,1);
        double aytmp[n];
        for(int i=0;i<n;i++)
        {
            aytmp[i] = a[i]*y;
        }
        double Xi1tmp[n];
        poiVecSub(1.0,Xi,n,Xi1tmp);
        double dxtmp[n];
        vecVecSub(zzxtmp,aytmp,n,dxtmp);
        vecVecSub(dxtmp,S,n,zzxtmp);
        vecVecSub(zzxtmp,Xi1tmp,n,dx);
        dy[0] = -vecvec_dot_mpi(a,X,n);
        
        for(int i=0;i<n;i++)
        {
            ds[i] = -S[i]*X[i];
        }
        poiVecSub(C,X,n,cxtmp);
        for(int i=0;i<n;i++)
        {
            dxi[i] = -Xi[i]*cxtmp[i];
        }
        double normdx=norm_mpi(r1,n,1);
        double normdy = abs(r2[0]);
        if(rank==0)
        {
            printf("Iteration %d:mu=%.16lf, norm(dx)=%.16lf, norm(dy)=%.16lf\n",iter,mu,normdx,normdy);
        }
        if(mu<0.0000001&&normdx<0.0000001&&normdy<0.0000001)
        {
            if(rank==0)
            {
                printf("Converged!\n");
            }
            return;
        }
        if(iter>200)
        {
            if(rank==0)
            {
                printf("Excceeds maximum iteration count 200! Abort.\n");
            }
            return;
        }
        poiVecSub(C,X,n,cxtmp);
        double D[n],maxD,minD,aD,iD;
        D[0] = S[0]/X[0]+Xi[0]/cxtmp[0];
        aD=D[0];
        iD=D[0];
        for(int i=1;i<n;i++)
        {
            D[i] = S[i]/X[i]+Xi[i]/cxtmp[i];
            if(aD<D[i])
            {
                aD=D[i];
            }
            if(iD>D[i])
            {
                iD=D[i];
            }
        }
        MPI_Allreduce(&aD, &maxD, 1, MPI_DOUBLE, MPI_MAX,MPI_COMM_WORLD);
        MPI_Allreduce(&iD, &minD, 1, MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);
        if(maxD/minD>1.0e16)
        {
            if(rank==0)
            {
                printf("D is too ill-conditioned! Terminating...");
            }
            return;
        }
        if(rank==0)
        {
            printf("max(D)/min(D)=%lf\n",maxD/minD);
        }
        double dztmp[n*d];
        diagScalarMat(D,Z,n,d,dztmp);
        double M[d*d];
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,d,d,n,1.0,Z,d,dztmp,d,0,M,d);
        double N[d*d];
        MPI_Reduce(M,N,d*d,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
        if(rank==0)
        {
            for(int i=0;i<d*d;i=i+d+1)
            {
                N[i]=N[i]+1.0;
                //N=Factorize(N)
            }
            LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'U',d,N,d);
        }
        
        NewtonStep(Z,D,N,C,a,X,S,Xi,r1,r2,r3,r4,n,d);
        double alpha = 1.0;
        for(int i=0;i<n;i++)
        {
            if(dx[i]<0)
            {
                alpha=min(alpha,-X[i]/dx[i]);
            }
            else if(dx[i]>0)
            {
                alpha=min(alpha,(C-X[i])/dx[i]);
            }
            if(ds[i]<0)
            {
                alpha = min(alpha,-S[i]/ds[i]);
            }
            if(dxi[i]<0)
            {
                alpha = min(alpha,-Xi[i]/dxi[i]);
            }
        }
        double alphatmp;
        MPI_Allreduce(&alpha,&alphatmp,1,MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);
        alpha = alphatmp;
        if(rank==0)
        {
            printf("alpha=%.16lf\n",alpha);
        }
        double mu_aff;
        double vectmp1[n],vectmp2[n];
        for(int i=0;i<n;i++)
        {
            vectmp1[i]=X[i]+alpha*dx[i];
            vectmp2[i]=S[i]+alpha*ds[i];
        }
        mu_aff=vecvec_dot_mpi(vectmp1,vectmp2,n);
        for(int i=0;i<n;i++)
        {
            vectmp1[i]=C-X[i]-alpha*dx[i];
            vectmp2[i]=Xi[i]+alpha*dxi[i];
        }
        mu_aff+=vecvec_dot_mpi(vectmp1,vectmp2,n);
        mu_aff/=(2*gn);
        if(rank==0)
        {
            printf("mu_aff=%.16lf\n",mu_aff);
        }
        double sigma = (mu_aff/mu)*(mu_aff/mu)*(mu_aff/mu);
        LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, 1, r1, 1, raff1, 1);
        LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, 1, r3, 1, raff3, 1);
        LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, 1, r4, 1, raff4, 1);
        raff2[0]=r2[0];
        double smtmp = sigma*mu;
        for(int i=0;i<n;i++)
        {
            ds[i]=smtmp-dx[i]*ds[i];
            dxi[i]=smtmp+dx[i]*dxi[i];
            dx[i]=0;
        }
        dy[0]=0;
        
        NewtonStep(Z,D,N,C,a,X,S,Xi,r1,r2,r3,r4,n,d);
        
        for(int i=0;i<n;i++)
        {
            r1[i] += raff1[i];
            r3[i] += raff3[i];
            r4[i] += raff4[i];
        }
        r2[0]+=raff2[0];
        alpha = 1000000.0;
        for(int i=0;i<n;i++)
        {
            if(dx[i]<0)
            {
                alpha=min(alpha,-X[i]/dx[i]);
            }
            else if(dx[i]>0)
            {
                alpha=min(alpha,(C-X[i])/dx[i]);
            }
            if(ds[i]<0)
            {
                alpha = min(alpha,-S[i]/ds[i]);
            }
            if(dxi[i]<0)
            {
                alpha = min(alpha,-Xi[i]/dxi[i]);
            }
        }
        MPI_Allreduce(&alpha,&alphatmp,1,MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);
        alpha = alphatmp;
        alpha*=0.99;
        if(rank == 0)
        {
            printf("Corrector alpha=%.16lf\n",alpha);
        }
        y+=alpha*dy[0];
        if(rank == 0)
        {
            printf("dy=%.16lf y=%.16lf\n",dy[0],y);
        }
        for(int i=0;i<n;i++)
        {
            X[i]+=(alpha*dx[i]);
            S[i]+=(alpha*ds[i]);
            Xi[i]+=(alpha*dxi[i]);
        }
        ++iter;
        /*
         if(rank == 0){
         for(int i=0;i<5;i++)
         {
         printf("X=%lf,S=%lf,Xi=%lf\n",X[i],S[i],Xi[i]);
         }
         }*/
    }
}

int length(double* a,double value,int n)
{
    int sum=0;
    for(int i=0;i<n;i++)
    {
        if(a[i]<value)
        {
            ++sum;
        }
    }
    return sum;
}

int lengthC(double* a,double value,double C,int n)
{
    int sum=0;
    for(int i=0;i<n;i++)
    {
        if((C-a[i])<value)
        {
            ++sum;
        }
    }
    return sum;
}

double g(double *X,double *Y,long long int n,int d,int ind,double *a,int *AS,int iAS,double gamma)
{
    double R = 0.0;
    for(int i=0;i<iAS;i++)
    {
        int j=AS[i];
        double xn[d];
        for(int k=0;k<d;k++)
        {
            xn[k]=X[k+j*d]-X[k+ind*d];
            //printf("%lf ",xn[k]);
        }
        //printf("\n");
        R+=Y[j]*a[j]*exp((-gamma)*normsquare(xn,1,d));
    }
    return R;
    
}

int readfileX(char* path,double* mat,int d)
{
    
    FILE *fp=fopen(path,"r");
    if(fp == NULL)
    {
        return 1;
    }
    char temp[10240];
    int i,j;
    i=0;j=0;
    //printf("1\n");
    while(!feof(fp))
    {
        i=0;
        //printf("2\n");
        fgets(temp,10240,fp);
        temp[strlen(temp)-1]= '\0';
        char *s;
        s=strtok(temp,",");
        
        mat[i+j*d] = atof(s);
        while(i<=d)
        {
            i++;
            //printf("s is %s %d\n",s,s[0]);
            s = strtok(NULL,",");
            if(s==NULL)
                break;
            mat[i+j*d]=atof(s);
            //printf("3\n");
        }
        
        j++;
    }
    
    return 1;
}

void transpose(double *a,int m,int n)
{
    double atmp[m*n];
    matTrans(a,m,n,atmp);
    for(int i=0;i<m*n;i++)
    {
        a[i] = atmp[i];
    }
}

void transposeQ(double *a,int m,int n)
{
    double qtmp[m*n];
    for(int i=0;i<m;i++)
    {
        for(int j=0;j<n;j++)
        {
            qtmp[j+i*n] = a[i+j*m];
        }
    }
    for(int i=0;i<m*n;i++)
    {
        a[i] = qtmp[i];
    }
}

int main(int argv, char *argc[])
{
    MPI_Init(&argv, &argc);
    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    double start, end, phase1start;
    if(rank == 0)
    {
        phase1start = MPI_Wtime();
    }
    if(argv < 8)
    {
        if(rank == 0)
            printf("ERROR : Too few arguments passed to run the program.\nRequired arguments :: <dataset filename(char*)> <no of records(int)> <no of featues(int)> <rank k(int)> <gamma(double)> <power refinement q(int)>\n");
        return 0;
    }
    if((np == 0) || ((np & (np - 1)) != 0))
    {
        if(rank == 0)
            printf("ERROR : no of processors should be in the power of 2, np : %d\n",np);
        return 0;
    }
    char *filename = argc[1];
    long long int gn = atoi(argc[2]);
    int d = atoi(argc[3]);
    int k = atoi(argc[4]);
    double gamma = strtod(argc[5], NULL);
    int q = atoi(argc[6]);
    double C = strtod(argc[7], NULL);
    if(rank == 0)
		printf("Running xSVM \n dataset - %s\n no of records - %d\n no of features - %d\n rank k - %d\n gamma - %f\n C - %f\n ower refinement q - %d\n",filename, gn, d, k, gamma, C, q);
    long long int lbegin = (gn * rank)/np;
    long long int lend = (gn * (rank + 1))/np;
    int ln = lend - lbegin;
    if (ln <= k)
    {
        if(rank == 0)
            printf("ERROR: ln %d must be larger than k %d; reduce np %d.\n", ln, k, np);
        return 0;
    }
    int i;
    float *X = (float*)malloc(sizeof(float) * d * ln);
    float *YY = (float*)malloc(sizeof(float) * ln);
    /* Read and distribute X and Y from file */
    start = MPI_Wtime();
    readtrainingfile(rank, np, filename, X, d, YY, d, gn);
    //read_and_dist(rank, np, "X.csv", X, d, YY, d, gn);
    end = MPI_Wtime();
    printf("rank %d :: main :: Time taken to read and distribute X and Y from a file is %f seconds\n", rank, end - start);
    float *A = (float*) malloc(sizeof(float) * ln * k);
    float *R = (float*) malloc(sizeof(float) * k * k);
    float *Q = (float*) malloc( sizeof(float) * ln * k);
    for (i = 0; i < ln*k; i++)
        A[i] = gaussrand();
    /* A = KA (A is a randomly generated)*/
    char Aname[10], Qname[10];
    start = MPI_Wtime();
    kernel_matmul(np, rank, gn, d, k, X, d, YY, gamma, A, ln); 
   // kernel_matmul2( gn, d, k, X, d, YY, gamma, A, ln);
    end = MPI_Wtime();
    printf("rank %d :: main :: Time taken to perform kernel matmul A=K*A is %f seconds\n", rank, end - start);
    /* A = AR */
    memcpy(Q, A, sizeof(float)*ln*k);
    start = MPI_Wtime();
    qr(gn, k, A, ln, R);
    MPI_Bcast(R, k*k, MPI_FLOAT, 0, MPI_COMM_WORLD);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, k, k, 1.0, A, ln, R, k, -1.0, Q, ln);
    float maqr = LAPACKE_slange(LAPACK_COL_MAJOR, 'M', ln, k, Q, ln);
    printf("R[%d]: max(A-QR)=%.16f\n", rank, maqr);
    end = MPI_Wtime();
    printf("rank %d :: main :: Time taken to perform qr A=A*R is %f seconds\n", rank, end - start);
    memcpy(Q, A, sizeof(float)* ln * k);
    float *CC = (float*) malloc( sizeof(float) * k * k );
    /* Q = KQ */
    if(q > 0)
    {
		start = MPI_Wtime();
    	for(i=0; i<q; ++i)
    	{
        	kernel_matmul(np, rank, gn, d, k, X, d, YY, gamma, Q, ln);
        	qr(gn, k, Q, ln, R);
    	}
    	end = MPI_Wtime();
    	printf("rank %d :: main :: Time taken to perform Q=K*Q for %d iterations is %f seconds\n", rank,q, end - start);
    }
    /* C = Q'KQ = A'* Q , where Q=KQ*/
    start = MPI_Wtime();
    memcpy(A, Q, sizeof(float)* ln * k);
    kernel_matmul(np, rank, gn, d, k, X, d, YY, gamma, A, ln);
    inner_product(rank, ln, k, Q, ln, A, ln, CC, k);
    end = MPI_Wtime();
    printf("rank %d :: main :: Time taken to perform the inner product C=Q'*K*Q is  %f seconds\n", rank, end - start);
    /* ||K - QCQ'|| */
    float fn = 0.0;
    start = MPI_Wtime();
    float *QC = (float*) malloc(sizeof(float) * ln * k);
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ln, k, k, 1.0, Q, ln, CC, k, 0.0, QC, ln);
    float normK;
    fn  = fnorm(np, rank, gn, d, k, QC, ln, Q, ln, X, d, YY, gamma, &normK);
    end = MPI_Wtime();
    if (rank==0) printf("rank %d :: main :: fnorm is %.16f\n, original norm=%.16f, ratio=%e\n", rank, fn, normK, fn/normK);
    start = MPI_Wtime();
    
    if(isnan(fn) || isnan(normK))
    	return 0;

    int myid = rank;
    int procnum = np;
    int constd = d;
    int constk = k;
    d = k;
    int n = ln;

    double *gC = (double*)malloc(sizeof(double) * d * d);
    convertfloat2double(d, d, CC, d, gC, d);
    double *gQ = (double*) malloc( sizeof(double) * n * d);
    convertfloat2double(d, n, Q, n, gQ, n); 
    transposeQ(gQ,n,d);
    MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM,MPI_COMM_WORLD);
    printf("File reading done. size(Q)=%ld,%d,size(C)=%d,%d\n",n,k,k,k);
    printf("gn=%ld,d=%d\n",gn,d);
   
    double *gX=(double*)malloc(sizeof(double)* gn * constd); 
    double *gY = (double*)malloc(sizeof(double)* gn);
    readfileY("Y.csv", gY);
    readfileX("X.csv", gX, constd);
    
    long long int rowb = (long long int)floor((myid * gn)/procnum);
    long long int rowe = (long long int)floor(((myid+1) * gn)/procnum);
    n = rowe-rowb;
    printf("R[%d] Q size %d,%d\n",myid, n, d);
    double gCTrans[d*d];
    matTrans(gC,d,d,gCTrans);
    matAdd(gC,gCTrans,d,d);
    int INFO = 0;
   
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',d,gC,d);
    setUTri2Zero(gC,d);
    
    double Qtmp[n*d];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,n,d,d,1.0,gQ,d,gC,d,0,Qtmp,d);
    LAPACKE_dlacpy(LAPACK_ROW_MAJOR,'A',n,d,Qtmp,d,gQ,d);
    double Y[n];
    LAPACKE_dlacpy(LAPACK_ROW_MAJOR, 'A', n, 1, gY+rowb-1, 1, Y, 1);
    double a[n],xi[n];
    MPC(gQ,Y,C,gamma,k,n,d,a,xi);
    
    if(myid==0)
    {
        printf("\nChecking feasibility...\n");
    }
    for(int i=0;i<n;i++)
    {
        if(a[i]<0||a[i]>C)
        {
            printf("R[%d]:Infeasible point! a[%d]=%lf\n",myid,i,a[i]);
        }
    }
    int a0,aC,atmp;
    a0=length(a,0.1,n);
    aC=lengthC(a,0.1,C,n);
    MPI_Allreduce(&a0,&atmp,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    a0=atmp;
    MPI_Allreduce(&aC,&atmp,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    aC=atmp;
    if(myid==0)
    {
        printf("#a[i]=0:%d\n",a0);
        printf("#a[i]=C:%d\n",aC);
    }
    double ga[gn],gxi[gn];
    fill(ga,0.0,gn);
    fill(gxi,0.0,gn);
    int counts[procnum];
    int displs[procnum];
    displs[0]=0;
    for(int i=0;i<procnum;i++)
    {
        counts[i] = (int)floor(i*gn*1.0/(procnum*1.0))-(int)floor((i-1)*gn*1.0/(procnum*1.0));
        
    }
    for(int i=1;i<procnum;i++)
    {
        displs[i]=displs[i-1]+counts[i-1];
    }
    MPI_Gatherv(a,n,MPI_DOUBLE,ga,counts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Gatherv(xi,n,MPI_DOUBLE,gxi,counts,displs,MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(myid==0)
    {
        printf("\n%lf %lf %d %d %d\nGenerating model file...\nReading the original data file...\n",gamma,C,k,n,d);
        int S[gn],AS[gn];
        int iS,iAS;
        iS=0;
        iAS=0;
        for(int i=0;i<gn;i++)
        {
            if(ga[i]>0.001*C&&ga[i]<0.999*C)
            {
                S[iS++]=i;
            }
            if(ga[i]>0.001*C)
            {
                AS[iAS++]=i;
            }
        }
        printf("Constructing model...\n");
        printf("Boundary SV (0<x<C)    #:%d\n",iS);
        printf("Non-Boundary SV (x>0)  #:%d\n",iAS);
        double bs[iS];
        fill(bs,0.0,iS);
        for(int i=0;i<iS;i++)
        {
            int j=S[i];
            bs[i] = gY[j]-g(gX,gY,gn,constd,j,ga,AS,iAS,gamma);
        }
        mat2csv(iS,1,bs,1,"bs.csv");
        double b;
        if(iS>0)
        {
	    printf("boundary SV > 0\n");
            b=vecSum(bs,iS)/iS;
            printf("Average(bs)=%lf\n",b);
	    printf("Intercept b=%lf\n",b);
	    printf("Now writing solution to model.csv...\n");
	    writeModel(constd, gX, "model.csv",b,gamma,ga,gY,AS,iAS);
        }
        else if (iAS>0)
        {
            printf("Boundary SV is empty!\n");
	    #define MIN(x,y) (((x)<(y))? (x) : (y))
	    double bs2[MIN(iAS,100)];
	    for(int i=0; i<MIN(iAS,100); i++)
	    {
		int j=AS[i];
		bs2[i] = gY[j]*(1-gxi[j]) - g(gX,gY,gn,constd,j,ga,AS,iAS,gamma);
		printf("[DEBUG:Model] i=%d,j=%d,bs2[i]=%e\n", i, j, bs2[i]);
	    }
	    b = vecSum(bs2,MIN(iAS,100)) / MIN(iAS,100);
	    double acc = 0;
	    for(int i=0; i<MIN(iAS,100); i++)
		acc += (bs2[i]-b)*(bs2[i]-b);
	    acc = sqrt(acc) / MIN(iAS,100);
	    printf("Average(bs)=%e, stddev(bs)=%e, sample=%d\n",b,acc, MIN(iAS,100) );
            printf("Intercept b=%e\n",b);
	    printf("Now writing solution to model.csv...\n");
	    writeModel(constd, gX, "model.csv",b,gamma,ga,gY,AS,iAS);
        }
	else
	{
	     printf("No support vectors! No model generated.\n");
	}    
    }
    end = MPI_Wtime();
    if(myid == 0)
    {
        printf("Time taken to perform phase 1 is %lf\n",start-phase1start);
        printf("Time taken to perform phase 2 is %lf\n",end-start);
    }
    //free(X); free(Y); free(R); free(A);
    MPI_Finalize();
    return 0;
}
