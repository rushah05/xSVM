#include <stdlib.h>
#include <math.h>
#define EPSILON 0.000001

// read training dataset
// X is (distributed) d * n
// Y is (distributed) n
void readtrainingfile(int rank, int np, char *filename, float *X, int ldX, float *Y, int d, long long int n)
{
	int rk = 0 , ln = 0;
	long long int lbegin = 0, lend = 0;
	if(rank == 0)
        {
                FILE *f = fopen(filename, "r");
                if(!f)
                {
		    exit(EXIT_FAILURE);
		}
		
		char *line;
    		ssize_t read;
    		size_t len = 0;
    		int row = 0;
                int i = 0;
		float *Xbuf;
		float *Ybuf;
    		for(int iter = 0; iter < n; ++iter)
		{
			if((read = getline(&line, &len, f)) != -1)	
			{
				if(lend == lbegin)
                        	{
                                	lbegin = (n * rk)/np;
                                	lend = (n * (rk + 1))/np;
                                	ln = lend -lbegin;
                                	Xbuf = (float*)malloc(sizeof(float) * d * ln);
					Ybuf = (float*)malloc(sizeof(float) * ln);
					i = 0;
                        	}

				char *label = strtok (line," ");
				char *featureset = strtok (NULL," ");
                       	 	Ybuf[i] = atof(featureset);
				while (featureset != NULL)
        			{
            				int counter = strlen(featureset);
					int foundcolon = 0;
					char *tempidx = (char*)malloc(sizeof(char));
					char *tempval = (char*)malloc(sizeof(char));
					int idxcounter = 0, valcounter = 0;
					for(int loop = 0; loop < counter; ++loop)
					{	
						if(featureset[loop] == ':')
						{
							++foundcolon;
							continue;
						}
						else
						{
							if(foundcolon == 0)
							{
								tempidx[idxcounter] = featureset[loop];
								++idxcounter;
								tempidx = (char*)realloc(tempidx, sizeof(char) * (idxcounter+1));			
							}
							else
							{
								tempval[valcounter] = featureset[loop];
								++valcounter;
								tempval = (char*)realloc(tempval, sizeof(char) * (valcounter+1));
							}
						}
					}
					tempidx[idxcounter] = '\0';
					tempval[valcounter] = '\0';
					int index = atoi(tempidx);
					float val = atof(tempval);
					Xbuf[index + (i*d)] = val;
					featureset = strtok (NULL, " ");
        			}
				++i;
				++lbegin;
				if(lbegin == lend)
                        	{
                                	if(rk == 0)
                                	{
                                      		LAPACKE_slacpy(LAPACK_COL_MAJOR,'P', d, ln, Xbuf, d, X, ldX);
                                      		cblas_scopy(ln, Ybuf, 1, Y, 1);
                                	}
                                	else
                                	{
                                     		MPI_Send(Xbuf, (d * ln), MPI_FLOAT, rk, 0, MPI_COMM_WORLD);
                                      		MPI_Send(Ybuf, ln, MPI_FLOAT, rk, 1, MPI_COMM_WORLD);
                                	}
                                	free(Xbuf);
					free(Ybuf);
                                	++rk;
                        	}
			}
		}
                fclose(f);
        }
	else
    	{
		MPI_Status status;
        	lbegin = (n * rank)/np;
        	lend = (n * (rank + 1))/np;
        	ln = lend - lbegin;
        	MPI_Recv(X, (d * ln), MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(Y, ln, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &status);
    	}
}

void convertfloat2double(int n, int d, float *F, int ldF, double *D, int ldD)
{
	for(int i=0; i<n; ++i)
	{
		for(int j=0; j<d; ++j)
		{
			D[j + i * ldD] = (double) F[j + i * ldF];
		}
	}
}


void isEqualMatrix(int m, int n, double *A, int ldA, double *B, int ldB)
{
    int i, j;
    for(j = 0; j < n; ++j)
    {
        for(i = 0; i < m; ++i)
        {
            if(abs(A[i + j * ldA] - *(B + i + j * ldB)) < EPSILON)
                printf(". ");
            else
                printf("x ");
        }
        printf("\n");
    }
}
