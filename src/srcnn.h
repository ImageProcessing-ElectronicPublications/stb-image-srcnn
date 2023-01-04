/* Program  : Image Super-Resolution using deep Convolutional Neural Networks
 * Author   : Wang Shu
 * Date     : Sun 13 Sep, 2015
 * Descrip. :
* */

#ifndef SRCNN_H_
#define SRCNN_H_

#ifdef SRCNN_STATIC
#define SRCNNAPI static
#else
#define SRCNNAPI extern
#endif

#ifdef __cplusplus
extern "C" {
#endif
SRCNNAPI void RGBtoYCbCrFilter(unsigned char *src, int height, int width, int channels, int direct);
SRCNNAPI void Convolution99x11x55(unsigned char *src, float* cnndst, int height, int width, int channels, float partcnn);
#ifdef __cplusplus
}
#endif

#ifdef SRCNN_IMPLEMENTATION

#include "convdata.h"

/* Marco Definition */
#define UP_SCALE 2

void RGBtoYCbCrFilter(unsigned char *src, int height, int width, int channels, int direct)
{
    int i, j, Y, cb, cr, r, g, b;
    size_t k;

    if (channels < 3) return;
    k = 0;
    if (direct < 0)
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                Y = (int)src[k + 0];
                cb = (int)src[k + 1];
                cr = (int)src[k + 2];
                r = Y + 1.402f * (cr - 128);
                g = Y - 0.344136f * (cb - 128) - 0.714136f * (cr - 128);
                b = Y + 1.772f * (cb - 128);
                src[k + 0] = (r < 0) ? 0 : (r < 255) ? r : 255;
                src[k + 1] = (g < 0) ? 0 : (g < 255) ? g : 255;
                src[k + 2] = (b < 0) ? 0 : (b < 255) ? b : 255;
                k += channels;
            }
        }
    }
    else
    {
        for (i = 0; i < height; i++)
        {
            for (j = 0; j < width; j++)
            {
                r = src[k + 0];
                g = src[k + 1];
                b = src[k + 2];
                Y = 0.299f  * r + 0.587f * g + 0.114f * b;
                cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;
                cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
                src[k + 0] = (Y < 0) ? 0 : (Y < 255) ? Y : 255;
                src[k + 1] = (cb < 0) ? 0 : (cb < 255) ? cb : 255;
                src[k + 2] = (cr < 0) ? 0 : (cr < 255) ? cr : 255;
                k += channels;
            }
        }
    }
}

/***
 * FuncName  : Convolution99x11x55
 * Function  : Complete one cell in the Convolutional Layer
 * Parameter : src - the original input/output image
 *             cnndst - the CNN (width * height * CONV2_FILTERS)
 *             kernel - the convolutional kernel
 *             bias - the cell bias
 * Output   : <void>
***/
void Convolution99x11x55(unsigned char *src, float* cnndst, int height, int width, int channels, float partcnn)
{
    int row, col, i, j, k, m, n;
    float temp[CONV1_FILTERS], result, t;
    int rowf, colf;
    double tp;
    size_t a;

    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            for (k = 0; k < CONV1_FILTERS; k++)
            {
                temp[k] = 0.0f;
                for (i = 0; i < 9; i++)
                {
                    rowf = row - 4 + i;
                    rowf = (rowf < 0) ? 0 : (rowf < height) ? rowf : (height - 1);
                    for (j = 0; j < 9; j++)
                    {
                        colf = col - 4 + j;
                        colf = (colf < 0) ? 0 : (colf < width) ? colf : (width - 1);
                        temp[k] += weights_conv1_data[k][i][j] * src[(rowf * width + colf) * channels];
                    }
                }
                temp[k] += biases_conv1[k];
                temp[k] = (temp[k] < 0.0f) ? 0.0f : temp[k];
            }
            for (k = 0; k < CONV2_FILTERS; k++)
            {
                result = 0.0f;
                for (i = 0; i < CONV1_FILTERS; i++)
                {
                    result += temp[i] * weights_conv2_data[k][i];
                }
                result += biases_conv2[k];
                result = (result < 0.0f) ? 0.0f : result;
                cnndst[(k * height + row) * width + col] = result;
            }
        }
    }
    a = 0;
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            t = 0.0f;
            for (i = 0; i < CONV2_FILTERS; i++)
            {
                tp = 0.0;
                for (m = 0; m < 5; m++)
                {
                    rowf = row - 2 + m;
                    rowf = (rowf < 0) ? 0 : (rowf < height) ? rowf : (height - 1);
                    for (n = 0; n < 5; n++)
                    {
                        colf = col - 2 + n;
                        colf = (colf < 0) ? 0 : (colf < width) ? colf : (width - 1);
                        tp += weights_conv3_data[i][m][n] * cnndst[(i * height + rowf) * width + colf];
                    }
                }
                t += (float)tp;
            }
            t += biases_conv3;
            t *= partcnn;
            t += ((1.0f - partcnn) * (float)src[a]);
            t += 0.5f;
            t = (t < 0.0f) ? 0.0f : (t < 255.0f) ? t : 255.0f;

            src[a] = (unsigned char)t;
            a += channels;
        }
    }

    return;
}

#endif /* SRCNN_IMPLEMENTATION */

#endif /* SRCNN_H_ */
