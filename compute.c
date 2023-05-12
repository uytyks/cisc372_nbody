#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
//we shall define 3 functions here


__global__ void arraySet(vector3* v, vector3** a){
        int in = threadIdx.x;
        for (i=in;i<NUMENTITIES;in += blockDim.x)
                a[i]=&v[i*NUMENTITIES];
}

__global__ void accelComp(vector3** a, vector3* pos, float* mass){
        int in = threadIdx.x;
        for (i=0;i<NUMENTITIES;i++){
                for (j=0;j<NUMENTITIES;j++){
                        if (i==j) {
                                FILL_VECTOR(accels[i][j],0,0,0);
                        }
                        else{
                                vector3 distance;
                                for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
                                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                                double magnitude=sqrt(magnitude_sq);
                                double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
                                FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
                        }
                }
        }
}

__global__ void sumMatrix(vector3** a, vector3* pos, vector3* vel){
        for (i=0;i<NUMENTITIES;i++){
                vector3 accel_sum={0,0,0};
                for (j=0;j<NUMENTITIES;j++){
                        for (k=0;k<3;k++)
                                accel_sum[k]+=accels[i][j][k];
                }
                //compute the new velocity based on the acceleration and time interval
                //compute the new position based on the velocity and time interval
                for (k=0;k<3;k++){
                        hVel[i][k]+=accel_sum[k]*INTERVAL;
                        hPos[i][k]=hVel[i][k]*INTERVAL;
                }
        }
}
void compute(){
        //make an acceleration matrix which is NUMENTITIES squared in size;
        int i,j,k;
        vector3* values;
        vector3** accels;
        double* d_mass;
        //device memory start
        //vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        //vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
        cudaMalloc(&values,sizeof(vector3*)*NUMENTITIES*NUMENTITIES);
        cudaMalloc(&accels,sizeof(vector3**)*NUMENTITIES);
        cudaMalloc(&d_mass,sizeof(float*));

        cudaMemcpy(d_hPos,hPos,sizeof(hPos),cudaMemcpyHostToDevice);
        cudaMemcpy(d_hVel,hVel,sizeof(hVel),cudaMemcpyHostToDevice);
        cudaMemcpy(d_mass,mass,sizeof(mass),cudaMemcpyHostToDevice);
        //for (i=0;i<NUMENTITIES;i++)
        //      accels[i]=&values[i*NUMENTITIES];
        arraySet<<<1,100>>>(values,accels);`
        //first compute the pairwise accelerations.  Effect is on the first argument.
        accelComp<<<1,100>>>(accels,d_hPos,d_mass);
        //sum up the rows of our matrix to get effect on each entity, then update velocity and position.
        sumMatrix<<<1,100>>>(accels,d_hPos,d_hVel);
        //free(accels);
        //free(values);
        //free device memory
        cudaFree(accels);
        cudaFree(values);
}
