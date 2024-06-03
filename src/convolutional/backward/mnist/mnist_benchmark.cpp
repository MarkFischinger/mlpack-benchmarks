#include <mlpack.hpp>
#include <chrono>

using namespace arma;
using namespace mlpack;
using namespace std;

Row<size_t> getLabels(const mat& predOut)
{
  Row<size_t> predLabels(predOut.n_cols);
  for (uword i = 0; i < predOut.n_cols; ++i)
  {
    predLabels(i) = predOut.col(i).index_max();
  }
  return predLabels;
}

int main()
{
  mlpack::math::RandomSeed(0);

  constexpr double RATIO = 0.1;
  
  const int EPOCHS = 1;

  const int BATCH_SIZE = 50;

  const double STEP_SIZE = 1.2e-3;

  cout << "Reading data ..." << endl;

  mat dataset;

  // The original file can be downloaded from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("./data/train.csv", dataset, true);

  mat train, valid;
  data::Split(dataset, train, valid, RATIO);


  const mat trainX = train.submat(1, 0, train.n_rows - 1, train.n_cols - 1) /
      256.0;
  const mat validX = valid.submat(1, 0, valid.n_rows - 1, valid.n_cols - 1) /
      256.0;

  const mat trainY = train.row(0);
  const mat validY = valid.row(0);

  FFN<NegativeLogLikelihood, RandomInitialization> model;

  model.Add<Convolution>(6,  
                         5,  
                         5,  
                         1,  
                         1,  
                         0,  
                         0  
  );

  model.Add<LeakyReLU>();

  model.Add<MaxPooling>(2,
                        2, 
                        2,
                        2, 
                        true);

  model.Add<Convolution>(16, 
                         5,  
                         5,  
                         1, 
                         1,  
                         0,  
                         0   
  );


  model.Add<LeakyReLU>();

  model.Add<MaxPooling>(2, 2, 2, 2, true);

  model.Add<Linear>(10);
  model.Add<LogSoftMax>();

  model.InputDimensions() = vector<size_t>({ 28, 28 });

  cout << "Start training ..." << endl;


  ens::Adam optimizer(
      STEP_SIZE,  
      BATCH_SIZE, 
                  
      0.9,        
      0.999, 
      1e-8,  
      EPOCHS * trainX.n_cols, 
      1e-8,           
      true);


    auto start = std::chrono::high_resolution_clock::now();
  model.Train(trainX,
              trainY,
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              ens::EarlyStopAtMinLoss(
                  [&](const arma::mat& )
                  {
                    double validationLoss = model.Evaluate(validX, validY);
                    cout << "Validation loss: " << validationLoss << "."
                        << endl;
                    return validationLoss;
                  }));

  auto end = std::chrono::high_resolution_clock::now();  
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  std::cout << "Time elapsed: " << duration.count() << "ms" << std::endl;

  mat predOut;
  model.Predict(trainX, predOut);

  Row<size_t> predLabels = getLabels(predOut);
  double trainAccuracy =
      accu(predLabels == trainY) / (double) trainY.n_elem * 100;

  model.Predict(validX, predOut);
  predLabels = getLabels(predOut);

  double validAccuracy =
      accu(predLabels == validY) / (double) validY.n_elem * 100;

  cout << "Accuracy: train = " << trainAccuracy << "%,"
            << "\t valid = " << validAccuracy << "%" << endl;

  data::Save("model.bin", "model", model, false);

  cout << "Predicting on test set..." << endl;

  // Get predictions on test data points.
  // The original file could be download from
  // https://www.kaggle.com/c/digit-recognizer/data
  data::Load("./data/test.csv", dataset, true);
  const mat testX = dataset.submat(1, 0, dataset.n_rows - 1, dataset.n_cols - 1)
      / 256.0;
  const mat testY = dataset.row(0);
  model.Predict(testX, predOut);

  predLabels = getLabels(predOut);
  double testAccuracy =
      accu(predLabels == testY) / (double) testY.n_elem * 100;

  cout << "Accuracy: test = " << testAccuracy << "%" << endl;

  cout << "Saving predicted labels to \"results.csv.\"..." << endl;
  predLabels.save("results.csv", arma::csv_ascii);

  cout << "Neural network model is saved to \"model.bin\"" << endl;

}