__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth, int imageChannels)
{
    //@@ Insert code to implement matrix multiplication here
    int i = get_global_id(0);
    int j = get_global_id(1);
    int maskRadius = (maskWidth / 2);
    for (int k = 0; k < imageChannels; k++)
    {
        float sum = 0;
        for (int y = -maskRadius; y <= maskRadius; y++)
        {
            for (int x = -maskRadius; x <= maskRadius; x++)
            {
                int x_offset = (j + x);
                int y_offset = (i + y);
                if (((x_offset >= 0) && (x_offset < width)) && ((y_offset >= 0) && (y_offset < height)))
                {
                    float imagePixel = inputData[(y_offset * width + x_offset) * imageChannels + k];
                    float maskValue = maskData[(y + maskRadius) * maskWidth + (x + maskRadius)];
                    sum += (imagePixel * maskValue);
                }
            }
        }
        if (sum < 0)
        {
            sum = 0;
        }
        else if (sum > 1)
        {
            sum = 1;
        }
        outputData[(i * width + j) * imageChannels + k] = sum;
    }
}