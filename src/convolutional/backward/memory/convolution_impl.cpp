template<
    typename ForwardConvolutionRule,
    typename BackwardConvolutionRule,
    typename GradientConvolutionRule,
    typename MatType
>
void ConvolutionType<
    ForwardConvolutionRule,
    BackwardConvolutionRule,
    GradientConvolutionRule,
    MatType
>::Backward(
    const MatType& /* input */,
    const MatType& /* output */,
    const MatType& gy,
    MatType& g)
{
  MatType gTemp(this->inputDimensions[0], this->inputDimensions[1],
                inMaps * higherInDimensions * batchSize, arma::fill::zeros);

  const bool usingPadding = (padWLeft || padWRight || padHTop || padHBottom);

  CubeType rotatedFilters(weight.n_rows, weight.n_cols, weight.n_slices);
  #pragma omp parallel for
  for (size_t map = 0; map < (size_t)(maps * inMaps); ++map)
    Rotate180(weight.slice(map), rotatedFilters.slice(map));

  CubeType dilatedMappedError;
  if (strideHeight > 1 || strideWidth > 1) {
    dilatedMappedError.set_size(mappedError.n_rows * strideWidth - (strideWidth - 1),
                                mappedError.n_cols * strideHeight - (strideHeight - 1),
                                mappedError.n_slices);
    dilatedMappedError.zeros();  
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < mappedError.n_slices; ++i)
      for (size_t j = 0; j < mappedError.n_cols; ++j)
        for (size_t k = 0; k < mappedError.n_rows; ++k)
          dilatedMappedError(k * strideWidth, j * strideHeight, i) = mappedError(k, j, i);
  } else {
    dilatedMappedError = mappedError;  
  }

  #pragma omp parallel for collapse(2)
  for (size_t offset = 0; offset < higherInDimensions * batchSize; ++offset)
    for (size_t inMap = 0; inMap < inMaps; ++inMap)
      for (size_t outMap = 0; outMap < maps; ++outMap)
        BackwardConvolutionRule::Convolution(
            dilatedMappedError.slice(outMap + offset * maps),
            rotatedFilters.slice((outMap * inMaps) + inMap),
            gTemp.slice(inMap + offset * inMaps),
            1, 1, 1, 1, true);

  if (usingPadding) {
    MatType gPadded(padding.OutputDimensions()[0], padding.OutputDimensions()[1],
                    inMaps * higherInDimensions, batchSize);
    paddingBackward.Forward(gTemp, gPadded);
    g = gPadded.tube(padWLeft, padHTop,
                     padWLeft + gTemp.n_rows - 1, padHTop + gTemp.n_cols - 1);
  } else {
    g = gTemp;  
  }
}