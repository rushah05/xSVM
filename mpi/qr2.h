#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <mpi.h>
// A is n * k (distributed)
// A should be overwriten by Q
// Algorithm: comm-avoiding Q
void stack(float *R1, float *R2, float *RR, int k);
void unstack(float *R1, float *R2, float *RR, int k);
void qr(int n, int k, float *A, int ldA, float *R)
{
    int rank, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ln = (rank+1)*n/np - (rank)*n/np; // local n
    if (ln <= k) printf("ERROR: ln %d must be larger than k %d; reduce np.\n", ln, k);
    
    float *tau = (float*)malloc(sizeof(float)*k);
    float *R1 = (float*)malloc(sizeof(float)*k*k); // k*k, my R
    float *R2 = (float*)malloc(sizeof(float)*k*k);
    float *RR = (float*)malloc(sizeof(float)*2*k*k);
    float *Q = (float*)malloc(sizeof(float)*2*k*k*log2(np)); // stores every Q.
    float *Q1 = (float*)malloc(sizeof(float)*k*k);
    float *Q2 = (float*)malloc(sizeof(float)*k*k);
    float *Qtmp = (float*)malloc(sizeof(float)*2*k*k); // tmp buf for DGEMM
    float *Atmp = (float*)malloc(sizeof(float)*2*ln*k);
    
    
    LAPACKE_sgeqrf(LAPACK_COL_MAJOR, ln, k, A, ldA, tau);
    int i,j;
    for (j=0; j<k; j++){
	for (i=0; i<k; i++) {
            R1[i+j*k] = A[i+j*ldA];
	    if (i>j) R1[i+j*k] = 0;
	}
    }
    
    /* float *tauQ = (float*)malloc(sizeof(float)*k*log2(np)); */
    LAPACKE_sorgqr(LAPACK_COL_MAJOR, ln, k, k, A, ldA, tau); // form Q explicitly in A.
    
    // forward pass;
    /*  example rank in binary:
     r=   0      1    2
     000  recv  recv  recv
     001  send
     010  recv  send
     011  send
     100  recv  recv  send
     101  send
     110  recv  send
     111  send
     */
    
    int r;
    MPI_Status status;
    /* printf("Forward pass...\n"); */
    for (r=0; r<log2(np); r++) {
        if (r== 0 || !( rank & ((1<<r)-1) )) {
            if ( !(rank & (1<<r))  ) { // recv
                /* printf("ROUND %d: R[%d]: Recv from %d\n", r, rank, rank ^(1<<r)); */
                MPI_Recv(R2, k*k, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD, &status);
                stack(R1, R2, RR, k);
                LAPACKE_sgeqrf(LAPACK_COL_MAJOR, 2*k, k, RR, 2*k, tau); // QR(RR);
                LAPACKE_slacpy(LAPACK_COL_MAJOR, 'U', k, k, RR, 2*k, R1, k); // R1=UPLO(RR);
                LAPACKE_sorgqr(LAPACK_COL_MAJOR, 2*k, k, k, RR, 2*k, tau); // form Q in QR(RR)
                LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2*k, k, RR, 2*k, &Q[2*k*k*r], 2*k); // store Q in Q[r]
            } else {
                /* printf("ROUND %d: R[%d]: Send to %d\n", r, rank, rank ^ (1<<r)); */
                MPI_Send(R1, k*k, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD);
            }
        }
    }
    /* MPI_Barrier(MPI_COMM_WORLD); */
    if (rank==0) {
        for (i=0; i<k; i++) {
            for (j=0; j<k; j++) {
                /* printf("%.6g ", R1[i+j*k]); */
                R[i+j*k] = R1[i+j*k];
                if (i>j) R[i+j*k] = 0;
            }
            /* printf(";"); */
        }
        printf("\n");
    }
    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* printf("R[0,0]=%f\n", R1[0]); */
    // backward pass: form Q
    /* MPI_Barrier(MPI_COMM_WORLD); */
    /* printf("Backward pass...\n"); */
    /* printf("Backward pass...\n"); */
    for (r=log2(np)-1; r>=0; r--) {
        if (r== 0 || !( rank & ((1<<r)-1) )) {
            if ( !(rank & (1<<r)) ) { // send
                /* printf("ROUND %d: R[%d]: Send\n", r, rank); */
                unstack(Q1, Q2, &Q[2*k*k*r], k); // [Q1; Q2] = Q
                MPI_Send(Q2, k*k, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD); // send Q2 to buddy
            } else {
                /* printf("ROUND %d: R[%d]: Recv\n", r, rank); */
                MPI_Recv(Q1, k*k, MPI_FLOAT, rank ^ (1<<r), 0, MPI_COMM_WORLD, &status); //recv Q1 from buddy
            }
            if (r>0) {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            2*k, k, k, 1.0, &Q[2*k*k*(r-1)], 2*k, Q1, k, 0, Qtmp, 2*k); // Q[r]*Q1
                LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', 2*k, k, Qtmp, 2*k, &Q[2*k*k*(r-1)], 2*k);
            }
        }
    }
    
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                ln, k, k, 1.0, A, ldA, Q1, k, 0, Atmp, ln);
    LAPACKE_slacpy(LAPACK_COL_MAJOR, 'A', ln, k, Atmp, ln, A, ldA);
    
    free(R1);free(R2); free(RR);
    free(tau);
    free(Q1); free(Q2); free(Q);
    free(Atmp); free(Qtmp);
}


void stack(float *R1, float *R2, float *RR, int k)
{
    int i,j;
    for (j=0; j<k; j++) {
        for (i=0; i<k; i++) {
            RR[i+j*(2*k)] = R1[i+j*k];
            RR[i+k+j*(2*k)] = R2[i+j*k];
        }
    }
}

void unstack(float *R1, float *R2, float *RR, int k)
{
    int i,j;
    for (j=0; j<k; j++) {
        for (i=0; i<k; i++) {
            R1[i+j*k] = RR[i+j*(2*k)];
            R2[i+j*k] = RR[i+k+j*(2*k)];
        }
    }
}


// for unit testing.
/*int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int n = 1234, k = 23;
    if (argc == 3) {
        n=atoi(argv[1]);
        k=atoi(argv[2]);
        //printf("setting n=%d, k=%d\n", n, k);
    }
    
    int rank, np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int length;
    char name[1000];
    MPI_Get_processor_name(name, &length);
    printf("rank %d on host %s\n", rank, name);
    int ln = (rank+1)*n/np - (rank)*n/np; // local n
    // printf("R[%d]: ln=%d k=%d\n", rank, ln, k);
    if (ln <= k) {
        printf("ERROR: ln %d must be larger than k %d; reduce np.\n", ln, k);
        return -1;
    }
    
    if (rank==0) {
        printf("QR(A) size %d*%d\n", n, k);
    }
    
    double *A = (double*)malloc(sizeof(double)*ln*k);
    double *Acopy = (double*)malloc(sizeof(double)*ln*k);
    int i,j;
    for (i=0; i<ln; i++) {
        for (j=0; j<k; j++) {
            A[i+j*ln] = (rank*ln+i) +j*rank;
            Acopy[i+j*ln] = A[i+j*ln];
        }
    }
    if (ln < 20) {
        printf("R[%d]:",rank);
        for(i=0; i<ln*k; i++) printf("%f ",A[i]);
        printf("\n");
    }
    double *R=(double*)malloc(sizeof(double)*k*k);
    MPI_Barrier(MPI_COMM_WORLD);
    double stime = MPI_Wtime();
    qr(n, k, A, ln, R);
    MPI_Barrier(MPI_COMM_WORLD);
    double dtime = MPI_Wtime() - stime;
    if (rank==0)
        printf("time %.2f (s)\n", dtime);
    MPI_Bcast(R, k*k, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("R[%d]: Q:",rank);
    // for(i=0; i<ln*k; i++) printf("%f ",A[i]);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                ln, k, k, 1.0, A, ln, R, k, -1, Acopy, ln);
    if (ln < 20) {
        printf("R[%d]:",rank);
        for(i=0; i<ln*k; i++) printf("%f ",Acopy[i]); printf("\n");
    }
    int bad = 0;
    for(i=0; i<ln*k; i++) {
        if( abs(Acopy[i]) > 0.001) {bad++;}
    }
    // if (bad ==0) {
    //     printf("R[%d] Sucess!\n", rank);
    // }    else  {
    //     printf("R[%d] Fail: %d\n", rank, bad);
    // }
    MPI_Barrier(MPI_COMM_WORLD);
    int totalbad=0;
    MPI_Reduce(&bad, &totalbad, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank ==0) {
        if (totalbad > 0)
            printf("R[%d]:FAILURE: totalbad=%d\n", rank, totalbad);
        else
            printf("R[%d]:SUCCESS!: totalbad=%d\n", rank, totalbad);
    }
    MPI_Finalize();
}*/
