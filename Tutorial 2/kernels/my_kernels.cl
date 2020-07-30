//Basic intensity histogram function that uses global variables

kernel void intensityHistogram(global const uchar* A, global int* B, global const int* C)
{
	//global id in 1D space
	int id = get_global_id(0);
	
	//get the pixel value
	int value = (int)A[id];

	//scales the result to pixel bin size and between 0 and 255 pixel values
	float result =((value)*C[0])/255;
	int scaled = round(result);

	//increment to the pixel value in the histogram
	atomic_inc(&B[scaled]);
}


//Intensity histogram function that uses local variables

kernel void intensityHistogramLocal(global const uchar* A, global int* B, global const int* C, local int* D, global const int* E)
{
	//global id in 1D space
	int id = get_global_id(0);
	//local id in 1D space
	int lid = get_local_id(0);
	
	//get the pixel value
	int value = (int)A[id];
	
	//scales the result to pixel bin size and between 0 and 255 pixel values
	float result =((value)*C[0])/255;
	int scaled = round(result);

	//wait for all of these to finish
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//increment to the pixel value in the local histogram segment
	atomic_inc(&D[A[id]]);

	//wait for all of these to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	//add the histogram segments together
	if (id < E[0])
	{
		atomic_add(&B[id], D[id]);
	}
}


//Cumulative histogram function that uses global variables (HS Scan)

kernel void cumulativeHistogram(global int* A, global int* B)
{
	//global id in 1D space
	int id = get_global_id(0);
	//buffer size
	int N = get_global_size(0);
	//create a global buffer to store data
	global int* C;
	
	barrier(CLK_GLOBAL_MEM_FENCE);

	//stride
	for (int stride=1; stride<N; stride*=2)
	{

		//if id is the same as or bigger than the stride
		if (id >= stride)
		{
			//add current and last a and store in b
			B[id] = A[id] + A[id - stride];
		}
		else
		{
			//copy a to b
			B[id] = A[id];
		}
		
		//wait for all of these to finish
		barrier(CLK_GLOBAL_MEM_FENCE);

		//copy over the data from buffer
		C = A;
		A = B;
		B = C;
	}
}


//Cumulative histogram function that uses global variables (Double Buffered HS Scan) 

kernel void cumulativeHistogramLocal(global const int* A, global int* B, local int* Local_A, local int* Local_B)
{
	//global id in 1D space
	int id = get_global_id(0);
	//local id in 1D space
	int lid = get_local_id(0);
	//buffer size
	int N = get_local_size(0);
	//create a local buffer to store data
	local int *Local_C;	

	//copy A from global to local memory
	Local_A[lid] = A[id];
	
	//wait for all of these to finish
	barrier(CLK_LOCAL_MEM_FENCE);

	//stride
	for (int stride = 1; stride < N; stride *= 2)
	{
		//if local id is the same as or bigger than the stride
		if (lid >= stride)
		{
			//add current and last a and store in b
			Local_B[lid] = Local_A[lid] + Local_A[lid - stride];
		}
		else
		{
			//copy a to b
			Local_B[lid] = Local_A[lid];
		}	

		//wait for all of these to finish
		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		Local_C = Local_B;
		Local_B = Local_A;
		Local_A = Local_C;
	}
	
	//copy to B buffer
	B[id] = Local_A[lid];
}


//Blelloch exclusive scan that uses global variables

kernel void cumulativeHistogram_bl(global int* A, global int* B)
{	
	//global id in 1D space
	int id = get_global_id(0);
	//buffer size
	int N = get_global_size(0);
	//create tmp variable for storage
	int t;
	   	 
	//sweep up
	for (int stride = 1; stride < N; stride *= 2)
	{
		//if 1 + id is exactly divisible by double stride
		if (((id + 1) % (stride*2)) == 0)
		{
			//add last stride
			A[id] += A[id - stride];
		}

		//wait for all of these to finish
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//sweep down
	
	//set last value to 0
	if (id == 0)
	{
		A[N-1] = 0;
	}
		
	//wait for all of these to finish
	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N/2; stride > 0; stride /= 2)
	{
		//if 1 + id is exactly divisible by double stride
		if (((id + 1) % (stride*2)) == 0)
		{
			//store current as tmp
			t = A[id];
			
			//add last stride
			A[id] += A[id - stride];

			//move tmp value
			A[id - stride] = t;
		}
		
		//wait for all of these to finish
		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//copy to B buffer
	B[id] = A[id];
}


//Equalised histogram function that uses global variables

kernel void equalisedHistogram(global int* A, global int* B)
{
	//global id in 1D space
	int id = get_global_id(0);
	//buffer size
	int N = get_global_size(0);

	//get current pixel and last pixel
	float a = (float)A[id];
	float an = (float)A[N-1];

	//equalise the pixel based on the max
	B[id] = (int)(a* 255 / (int)an);
}


//histogram to image function that uses global variables

kernel void backProjection(global int* A, global uchar* B, global uchar* C, global int* D)
{
	//global id in 1D space
	int id = get_global_id(0);

	//convert input pixel to int
	int val = (int)B[id];

	//calculate output pixel based on pixel range and pixel bin
	C[id] = A[val/(256/D[0])];
}


//Normalised histogram function that uses global variables

kernel void normaliseImage(global const uchar* A, global uchar* B, global const int* C, global const int* D)
{
	//global id in 1D space
	int id = get_global_id(0);

	//normalise
	int result = ((A[id]-C[0])*255)/(D[0]-C[0]);
	
	//convert output back to unsigned character
	B[id] = (uchar)result;
}