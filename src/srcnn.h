/*
https://github.com/rageworx/SRCNN_OpenCV_GCC/issues/7
*/

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
SRCNNAPI void SRCNNblock(unsigned char *src, unsigned char *block, float* cnndst, int height, int width, int channels, int bsize, float partcnn);
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

// offset >= 4 + 2 = 6 !!!
void Convolution99x11x55offset(unsigned char *src, float* cnndst, int height, int width, int offset, float partcnn)
{
    int row, col, rowo, colo, rowf, colf, i, j, k, m, n;
    float temp[CONV1_FILTERS], result, t;
    double tp;
    size_t li, ld;

    for (row = offset - 2; row < height + 2 - offset; row++)
    {
        for (col = offset - 2; col < width + 2 - offset; col++)
        {
            for (k = 0; k < CONV1_FILTERS; k++)
            {
                temp[k] = 0.0f;
                for (i = 0; i < 9; i++)
                {
                    rowf = row - 4 + i;
                    for (j = 0; j < 9; j++)
                    {
                        colf = col - 4 + j;
                        li = rowf * width + colf;
                        temp[k] += weights_conv1_data[k][i][j] * src[li];
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
                ld = (k * height + row) * width + col;
                cnndst[ld] = result;
            }
        }
    }
    for (row = offset; row < height - offset; row++)
    {
        for (col = offset; col < width - offset; col++)
        {
            t = 0.0f;
            for (i = 0; i < CONV2_FILTERS; i++)
            {
                tp = 0.0;
                for (m = 0; m < 5; m++)
                {
                    rowf = row - 2 + m;
                    for (n = 0; n < 5; n++)
                    {
                        colf = col - 2 + n;
                        ld = (i * height + rowf) * width + colf;
                        tp += weights_conv3_data[i][m][n] * cnndst[ld];
                    }
                }
                t += (float)tp;
            }
            li = row * width + col;
            t += biases_conv3;
            t *= partcnn;
            t += ((1.0f - partcnn) * (float)src[li]);
            t += 0.5f;
            t = (t < 0.0f) ? 0.0f : (t < 255.0f) ? t : 255.0f;
            src[li] = (unsigned char)t;
        }
    }

    return;
}

void SRCNNblock(unsigned char *src, unsigned char *block, float* cnndst, int height, int width, int channels, int bsize, float partcnn)
{
    int i, j, bd, bb, bo, bs2, bm, bn, x, y, x0, y0, xf, yf;
    size_t ki, k;

    // offset >= 4 + 2 = 6 !!!
    bd = 6;
    bb = bsize + bd + bd;
    bo = bsize / 16;
    bs2 = bsize - bo - bo;
    bm = (height + bs2 - 1) / bs2;
    bn = (width + bs2 - 1) / bs2;

    for (i = 0; i < bm; i++)
    {
        y0 = i * bs2 - bo;
        for (j = 0; j < bn; j++)
        {
            x0 = j * bs2 - bo;
            k = 0;
            for (y = 0; y < bb; y++)
            {
                yf = y0 + y - bd;
                yf = (yf < 0) ? 0 : (yf < height) ? yf : (height - 1);
                for (x = 0; x < bb; x++)
                {
                    xf = x0 + x - bd;
                    xf = (xf < 0) ? 0 : (xf < width) ? xf : (width - 1);
                    ki = (yf * width + xf) * channels;
                    block[k] = src[ki];
                    k++;
                }
            }
            Convolution99x11x55offset(block, cnndst, bb, bb, bd, partcnn);
            for (y = bo; y < bsize - bo; y++)
            {
                yf = y0 + y;
                if (yf < height)
                {
                    for (x = bo; x < bsize - bo; x++)
                    {
                        xf = x0 + x;
                        if (xf < width)
                        {
                            ki = (yf * width + xf) * channels;
                            k = ((y + bd) * bb + x + bd);
                            src[ki] = block[k];
                        }
                    }
                }
            }
        }
    }

    return;
}

#endif /* SRCNN_IMPLEMENTATION */

#endif /* SRCNN_H_ */
