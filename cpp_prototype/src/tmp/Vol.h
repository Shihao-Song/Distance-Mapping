#ifndef _VOL_H_
#define _VOL_H_

class Vol
{
public:
	Vol(int height, int width, int depth) : height(height), width(width), depth(depth)
	{
		raw_vol = (unsigned char *)malloc(height * width * depth * sizeof(unsigned char));
	
		for(int i = 0; i < height * width * depth; i++)
        	{
                	raw_vol[i] = 0;
        	}

		genFC();
	}

	~Vol()
	{
		free(raw_vol);
	}

	/*
		Volume size
	*/
	int height;
	int width;
	int depth;

	unsigned char *raw_vol;

	/*
		Spacing
	*/
	float sp[3] = {1.0, 2.5, 1.5}; // Voxel spacings, {i (height of a voxel), 
					//              j (width of a voxel), 
					//              k (depth of a voxel)}
        
	float sp2[3] = {
                sp[0] * sp[0],
                sp[1] * sp[1],
                sp[2] * sp[2]
	};

private:

	void genFC()
	{
        	
		/* Distance between slices */
        	int slice_stride = height * width;

        	int row_id, col_id, dep_id;

        	/* Front and back (of the feature cuboid) */
        	for (row_id = 1; row_id < (height - 1); row_id++)
        	{
                	for (col_id = 1; col_id < (width - 1); col_id++)
                	{
                        	// Front FVs should have negative K face
                        	raw_vol[0 * slice_stride + row_id * width + col_id] |= 0x04;

                        	// Back FVs should have positive K face
                        	raw_vol[(depth - 1) * slice_stride + row_id * width + col_id] |= 0x20;
                	}
        	}

        	/* Left and right */
        	for (dep_id = 0; dep_id < depth; dep_id++)
        	{
               		for (row_id = 1; row_id < (height - 1); row_id++)
                	{
                        	// Left FVs should have negative I face
                        	raw_vol[dep_id * slice_stride + row_id * width + 1] |= 0x01;

                        	// Right FVs should have positive I face
                        	raw_vol[dep_id * slice_stride + row_id * width + (width - 2)] |= 0x08;
                	}
        	}	

        	/* Top and bottom */
        	for (dep_id = 0; dep_id < depth; dep_id++)
        	{
                	for (col_id = 1; col_id < (width - 1); col_id++)
                	{
                        	raw_vol[dep_id * slice_stride + 1 * width + col_id] |= 0x02;

                        	raw_vol[dep_id * slice_stride + (height - 2) * width + col_id] |= 
												0x10;
                	}
        	}
	}
};

#endif
