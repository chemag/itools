# itools: A set of image-processing tools

itools is a repo containing a set of tools for image processing.

# 1. itools-bayer-conversion.py

This tool converts between different Bayer formats.


# 2. itools-filter.py

This tool provides some image filters, including:
* `gray`: convert image to grayscale.
* `xchroma`: swap chromas (Cb and Cr) in the image.
* `noise`: add noise to the image.
* `diff`: get the diff of 2x images.
* `compose`: compose 2x images.
* `match`: match 2x images (template matching).
* `affine`: performs an affine transform in the input image.


![Figure 1](docs/lena.jpeg)

Figure 1 shows an example of an image.


## 2.1. `gray` filter

This filter converts an image to grayscale.

Example
```
$ ./python/itools-filter.py --filter gray -i docs/lena.jpeg -o docs/lena.gray.jpeg
```

![Figure 2](docs/lena.gray.jpeg)

Figure 2 shows the original image after being passed through the `gray` filter.


## 2.2. `xchroma` filter

This filter swaps the chromas (Cb and Cr) in the input image.

Example
```
$ ./python/itools-filter.py --filter xchroma -i docs/lena.jpeg -o docs/lena.xchroma.jpeg
```

![Figure 3](docs/lena.xchroma.jpeg)

Figure 3 shows the original image after being passed through the `xchroma` filter.


## 2.3. `noise` filter

This filter adds noise to the input image. The parameter "`--noise-level`" can
be used to add more or less noise.

Example
```
$ ./python/itools-filter.py --filter noise -i docs/lena.jpeg -o docs/lena.noise.jpeg
```

![Figure 4](docs/lena.noise.jpeg)

Figure 4 shows the original image after being passed through the `noise` filter.


## 2.4. `diff` filter

This filter gets the difference between 2x frames.

The diff algo works as follows: We convert both frames to grayscale (luma-only), and then diff the actual value of each pixel. We calculate the absolute value of the per-pixel difference (`abs[i,j]`), and then set each pixel in the output file (the "diff image" or "diff frame") as 255 minus the absolute value.

The full algo is:

```
# start with in1[i, j] and in2[i, j]
abs[i,j] = abs(in1[i, j], in2[i, j])
out[i,j] = 255 - abs[i,j]
```

Note that the parts where both input file are different is shown in black, while the parts where they are the same are shown in white.


Example
```
$ ./python/itools-filter.py --filter diff -i docs/lena.jpeg -j docs/lena.noise.jpeg -o docs/lena.diff.jpeg
```

![Figure 5](docs/lena.diff.jpeg)

Figure 5 shows the diff between the original image and the output of the `noise` filter.


## 2.5. `compose` filter

This filter composes 2x images, a background image and a needle image. It uses the needle image's alpha channel if it has one. The parameters "`-x`" and "`-y`" can be used to decide the exact location ((0,0) being the top-left point in the destination image).

Example
```
$ ./python/itools-filter.py --filter compose -i docs/lena.jpeg -j docs/needle.png -x 10 -y 30 -o docs/lena.compose.jpeg
```

![Figure 6](docs/lena.compose.jpeg)

Figure 6 shows the original image after being composed with the needle image.


## 2.6. `match` filter

This filter performs template matching in 2x images, a haystack image and a needle image. It will produce a copy of the haystack image with a rectangle marking the actual location where it finds the needle in the haystack.

Note that, if the needle has alpha channel, we use randomness to deal with needles with alpha channel (see [here](https://stackoverflow.com/questions/4761940/) for a discussion). This means that, in that case, the results of successive runs may be slightly different.


Example
```
$ ./python/itools-filter.py --filter match -d -i docs/lena.compose.jpeg -j docs/needle.png -o docs/lena.match.jpeg
x0 = 10 y0 = 30
```

![Figure 7](docs/lena.match.jpeg)

Figure 7 shows the original image after being composed with the needle image.



## 2.7. `affine` filter

This filter performs an affine transformation on the input image. The affine transformation is defined using 2x matrices, $A$ and $B$.

```
A = [[a00, a01], [a10, a11]]
B = [[b00], [b10]]
Output = A * input + B
```

Function also allows defining the output size (using parameters "`width`" and "`height`").

Example
```
$ ./python/itools-filter.py --filter affine --height 700 --a00 0.98 --a01 0.14 --a10 1.1 --a11 -0.3 --b00 1 --b10 10 -d -i docs/lena.jpeg -o docs/lena.affine.jpeg
```

![Figure 8](docs/lena.affine.jpeg)

Figure 8 shows the output of the affine transformation using the matrices $A=[[0.98, 0.14], [1.1, -0.3])$ and $B=[1, 10]$.


## 2.8. `affine-points` filter

This filter performs an affine transformation on the input image. The affine transformation is defined using 2x set of points, $s$ and $d$. Transformation matrix is calculated such that

```
(s0x, s0y) -> (d0x, d0y)
(s1x, s1y) -> (d1x, d1y)
(s2x, s2y) -> (d2x, d2y)
```

Function also allows defining the output size (using parameters "`width`" and "`height`").

Example
```
$ ./python/itools-filter.py --filter affine-points --height 700 --d0x 5 --d0y -2 --d1x -4 --d1y 105 --d2x 96 --d2y -4 -d -i docs/lena.jpeg -o docs/lena.affine-points.jpeg
...
transform_matrix = array([[ 0.91, -0.09,  5.  ],
       [-0.02,  1.07, -2.  ]])
...
```

![Figure 9](docs/lena.affine-points.jpeg)

Figure 9 shows the output of the affine transformation using the points in the above command.


## 2.9. `rotate` filter

This filter rotates an image (0, 90, 180, or 270 degrees).

Example
```
$ ./python/itools-filter.py --filter rotate -i docs/lena.jpeg --rotate-angle -90 -o docs/lena.rotate.png
```

![Figure 10](docs/lena.rotate.png)

Figure 10 shows the original image after being passed through the `rotate` filter.


# 3. Requirements

* opencv2
* numpy
