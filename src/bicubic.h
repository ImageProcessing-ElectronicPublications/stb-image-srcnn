/*
https://github.com/heptagonhust/bicubic-image-resize/issues/9
*/

#ifndef BICUBIC_H_
#define BICUBIC_H_

#ifdef BICUBIC_STATIC
#define BICUBICAPI static
#else
#define BICUBICAPI extern
#endif

#ifdef __cplusplus
extern "C" {
#endif
BICUBICAPI void ResizeImageBiCubic(unsigned char *src, int height, int width, int channels, int resize_height, int resize_width, unsigned char *res);
#ifdef __cplusplus
}
#endif

#ifdef BICUBIC_IMPLEMENTATION

float BiCubicWeightCoeff(float x, float a)
{
    if (x <= 1.0f)
    {
        return (1.0f - ((a + 3.0f) - (a + 2.0f) * x) * x * x);
    }
    else if (x < 2.0f)
    {
        return ((-4.0f + (8.0f - (5.0f - x) * x) * x) * a);
    }
    return 0.0f;
}

void BiCubicCoeff4x4(float y, float x, float *coeff)
{
    const float a = -0.5f;

    float u = y - (int)y;
    float v = x - (int)x;

    u += 1.0f;
    v += 1.0f;

    int k = 0;
    for (int i = 0; i < 4; i++)
    {
        float du = (u > i) ? (u - i) : (i - u);
        for (int j = 0; j < 4; j++)
        {
            float dv = (v > j) ? (v - j) : (j - v);
            coeff[k] = BiCubicWeightCoeff(du, a) * BiCubicWeightCoeff(dv, a);
            k++;
        }
    }
}

void BiCubicFilter(unsigned char *pix, unsigned char *src, int height, int width, int channels, float y_float, float x_float)
{
    float coeff[16];
    float sum[8] = {0.0f};

    int y0 = (int)y_float - 1;
    int x0 = (int)x_float - 1;
    BiCubicCoeff4x4(y_float, x_float, coeff);

    size_t k = 0, l;
    for (int i = 0; i < 4; i++)
    {
        int yf = ((y0 + i) < 0) ? 0 : ((y0 + i) < height) ? (y0 + i) : (height - 1);
        for (int j = 0; j < 4; j++)
        {
            int xf = ((x0 + j) < 0) ? 0 : ((x0 + j) < width) ? (x0 + j) : (width - 1);
            l = (yf * width + xf) * channels;
            for (int d = 0; d < channels; d++)
            {
                sum[d] += coeff[k] * (float)src[l + d];
            }
            k++;
        }
    }
    for (int d = 0; d < channels; d++)
    {
        pix[d] = (unsigned char)((sum[d] < 0.0f) ? 0 : (sum[d] < 255.0f) ? sum[d] : 255);
    }
}

BICUBICAPI void ResizeImageBiCubic(unsigned char *src, int height, int width, int channels, int resize_height, int resize_width, unsigned char *res)
{
    float ratio_height = (float)resize_height / (float)height;
    float ratio_width = (float)resize_width / (float)width;
    unsigned char pix[8];

    size_t k = 0;
    for (int i = 0; i < resize_height; i++)
    {
        float src_y = ((float)i + 0.5f) / ratio_height - 0.5f;
        for (int j = 0; j < resize_width; j++)
        {
            float src_x = ((float)j + 0.5f) / ratio_width - 0.5f;
            BiCubicFilter(pix, src, height, width, channels, src_y, src_x);
            for (int d = 0; d < channels; d++)
            {
                res[k] = pix[d];
                k++;
            }
        }
    }
}

#endif /* BICUBIC_IMPLEMENTATION */

#endif /* BICUBIC_H_ */
