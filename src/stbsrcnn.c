#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
#include "bicubic.h"
#include "srcnn.h"

#define SRCNN_VERSION "1.2"
#define UP_SCALE 2
#define CONV1_FILTERS   64      // the first convolutional layer
#define CONV2_FILTERS   32      // the second convolutional layer

void srcnn_usage(char* prog, int bsize, int overlap, float pcnn)
{
    printf("StbSRCNN version %s.\n", SRCNN_VERSION);
    printf("usage: %s [options] image_in out.png\n", prog);
    printf("options:\n");
    printf("  -b NUM    block size (default %d)\n", bsize);
    printf("  -o NUM    overlap block (default %d)\n", overlap);
    printf("  -p N.N    specifies the part of the CNN in the result (default %f)\n", pcnn);
    printf("  -h        show this help message and exit\n");
}

int main(int argc, char **argv)
{
    int resize_height = 0, resize_width = 0;
    int bsize = 256, overlap = 16;
    float pcnn = 0.707107f;
    int fhelp = 0;
    int opt;
    while ((opt = getopt(argc, argv, ":b:o:p:h")) != -1)
    {
        switch(opt)
        {
        case 'b':
            bsize = atoi(optarg);
            if (bsize < 64)
            {
                fprintf(stderr, "ERROR: block size %d < 64\n", bsize);
                return 2;
            }
            break;
        case 'o':
            overlap = atoi(optarg);
            if (overlap < 0)
            {
                fprintf(stderr, "ERROR: overlap block %d < 0\n", overlap);
                return 2;
            }
            break;
        case 'p':
            pcnn = atof(optarg);
            break;
        case 'h':
            fhelp = 1;
            break;
        case ':':
            fprintf(stderr, "ERROR: option needs a value\n");
            return 2;
            break;
        case '?':
            fprintf(stderr, "ERROR: unknown option: %c\n", optopt);
            return 3;
            break;
        }
    }
    if(optind + 2 > argc || fhelp)
    {
        srcnn_usage(argv[0], bsize, overlap, pcnn);
        return 0;
    }
    const char *src_name = argv[optind];
    const char *dst_name = argv[optind + 1];

    int height, width, channels;

    printf("Load: %s\n", src_name);
    stbi_uc* img = NULL;
    if (!(img = stbi_load(src_name, &width, &height, &channels, STBI_rgb_alpha)))
    {
        fprintf(stderr, "ERROR: not read image: %s\n", src_name);
        return 1;
    }
    printf("image: %dx%d:%d\n", width, height, channels);
    unsigned char* data = NULL;
    if (!(data = (unsigned char*)malloc(height * width * channels * sizeof(unsigned char))))
    {
        fprintf(stderr, "ERROR: not use memmory\n");
        return 1;
    }
    size_t ki = 0, kd = 0;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int d = 0; d < channels; d++)
            {
                data[kd + d] = (unsigned char)img[ki + d];
            }
            ki += STBI_rgb_alpha;
            kd += channels;
        }
    }
    stbi_image_free(img);

    resize_height = height * UP_SCALE;
    resize_width = width * UP_SCALE;
    if ((resize_height == 0) || (resize_width == 0))
    {
        fprintf(stderr, "ERROR: bad target size %dx%d:%d\n", resize_width, resize_height, channels);
        return 1;
    }
    printf("resize: %dx%d:%d\n", resize_width, resize_height, channels);

    unsigned char *resize_data = NULL;
    if (!(resize_data = (unsigned char*)malloc(resize_height * resize_width * channels * sizeof(unsigned char))))
    {
        fprintf(stderr, "ERROR: not memmory for resize\n");
        return 2;
    }

    printf("method: bicubic\n");
    ResizeImageBiCubic(data, height, width, channels, resize_height, resize_width, resize_data);

    printf("color: YCbCr\n");
    RGBtoYCbCrFilter(resize_data, resize_height, resize_width, channels, 1);

    size_t cnn_size = (bsize + 2 * (overlap + 6)) * (bsize + 2 * (overlap + 6)); // 6 == offset!
    unsigned char *cnn_block = NULL;
    if (!(cnn_block = (unsigned char*)malloc(cnn_size * sizeof(unsigned char))))
    {
        fprintf(stderr, "ERROR: not memmory for CNN\n");
        return 2;
    }
    float *cnn_data = NULL;
    if (!(cnn_data = (float*)malloc(cnn_size * CONV2_FILTERS * sizeof(float))))
    {
        fprintf(stderr, "ERROR: not memmory for CNN\n");
        return 2;
    }

    printf("method: CNN(%dx%d)\n", bsize, bsize);
//    Convolution99x11x55(resize_data, cnn_data, resize_height, resize_width, channels, pcnn);
    SRCNNblock(resize_data, cnn_block, cnn_data, resize_height, resize_width, channels, bsize, overlap, pcnn);
    printf("part CNN: %f\n", pcnn);

    printf("color: RGB\n");
    RGBtoYCbCrFilter(resize_data, resize_height, resize_width, channels, -1);

    printf("Save png: %s\n", dst_name);
    if (!(stbi_write_png(dst_name, resize_width, resize_height, channels, resize_data, resize_width * channels)))
    {
        fprintf(stderr, "ERROR: not write image: %s\n", dst_name);
        return 1;
    }

    free(resize_data);
    free(data);
    return 0;
}
