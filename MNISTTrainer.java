import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * MNISTTrainer.java
 *
 * Combina um MNIST loader (lê arquivos .gz no formato IDX) com uma rede neural
 * feedforward
 * implementada em Java puro. A rede tem 4 camadas ocultas e usa sigmoid como
 * ativação.
 *
 * Uso:
 * javac MNISTTrainer.java
 * java MNISTTrainer <train-images.gz> <train-labels.gz> <test-images.gz>
 * <test-labels.gz> <epochs> <learningRate> <output-log-file> [maxTrainSamples]
 *
 * Exemplo:
 * java MNISTTrainer train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
 * t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz 20 0.1
 * training_output.txt 60000
 *
 * Parâmetros:
 * - arquivos .gz do MNIST (você pode usar os links que forneceu)
 * - epochs: número de épocas
 * - learningRate: taxa de aprendizagem (ex.: 0.1)
 * - output-log-file: arquivo onde será gravado "Epoch, MSE"
 * - maxTrainSamples (opcional): número máximo de amostras de treino a usar
 * (útil para testes rápidos)
 *
 * Observações:
 * - Treinamento com todas as 60.000 amostras pode demorar. Use maxTrainSamples
 * para reduzir.
 * - Implementação didática: SGD (atualização por amostra). Para
 * produção/eficiência, usar mini-batches/vetorização.
 * como rodar no terminal: 
 * java MNISTTrainer "C:\Users\sofia\Desktop\Treinamento de uma rede neural para reconhecer digitos\train-images-idx3-ubyte.gz" "C:\Users\sofia\Desktop\Treinamento de uma rede neural para reconhecer digitos\train-labels-idx1-ubyte.gz" "C:\Users\sofia\Desktop\Treinamento de uma rede neural para reconhecer digitos\t10k-images-idx3-ubyte.gz" "C:\Users\sofia\Desktop\Treinamento de uma rede neural para reconhecer digitos\t10k-labels-idx1-ubyte.gz" 10 0.01 "saida_treino.txt" 500
 */

public class MNISTTrainer {

    // -----------------------
    // MNIST loader (lê .gz IDX)
    // -----------------------
    static class MnistDataloader {
        private final String imagesGzPath;
        private final String labelsGzPath;

        public MnistDataloader(String imagesGzPath, String labelsGzPath) {
            this.imagesGzPath = imagesGzPath;
            this.labelsGzPath = labelsGzPath;
        }

        public static class DataSet {
            public final byte[][] images; // flattened images: each length = rows * cols
            public final int[] labels;
            public final int rows;
            public final int cols;

            public DataSet(byte[][] images, int[] labels, int rows, int cols) {
                this.images = images;
                this.labels = labels;
                this.rows = rows;
                this.cols = cols;
            }
        }

        // Lê IDX gz (imagens) e labels
        public DataSet load() throws IOException {
            // read labels
            int[] labels;
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(
                    new GZIPInputStream(new FileInputStream(labelsGzPath))))) {
                int magic = dis.readInt();
                if (magic != 2049)
                    throw new IOException("Labels file magic mismatch (expected 2049), got " + magic);
                int numLabels = dis.readInt();
                labels = new int[numLabels];
                for (int i = 0; i < numLabels; i++) {
                    labels[i] = dis.readUnsignedByte();
                }
            }

            // ler imagens
            byte[][] images;
            int rows, cols;
            try (DataInputStream dis = new DataInputStream(new BufferedInputStream(
                    new GZIPInputStream(new FileInputStream(imagesGzPath))))) {
                int magic = dis.readInt();
                if (magic != 2051)
                    throw new IOException("Images file magic mismatch (expected 2051), got " + magic);
                int numImages = dis.readInt();
                rows = dis.readInt();
                cols = dis.readInt();
                images = new byte[numImages][rows * cols];
                for (int i = 0; i < numImages; i++) {
                    int offset = 0;
                    while (offset < rows * cols) {
                        int r = dis.read(images[i], offset, rows * cols - offset);
                        if (r < 0)
                            throw new EOFException("Unexpected EOF while reading image bytes");
                        offset += r;
                    }
                }
            }

            if (images.length != labels.length) {
                throw new IOException(
                        "Number of images and labels do not match: " + images.length + " vs " + labels.length);
            }
            return new DataSet(images, labels, rows, cols);
        }
    }

    // -----------------------
    // Neural network (feedforward, sigmoid, 4 hidden layers)
    // -----------------------
    static class NeuralNetwork {
        int[] sizes; // sizes[0] = input, sizes[L-1] = output
        double[][] biases; // biases[l][i] for layer l
        double[][][] weights; // weights[l][i][j] weight from neuron j in l-1 to neuron i in l
        Random rnd;

        public NeuralNetwork(int[] sizes, long seed) {
            this.sizes = sizes;
            int L = sizes.length;
            biases = new double[L][];
            weights = new double[L][][];
            rnd = new Random(seed);

            for (int l = 1; l < L; l++) {
                biases[l] = new double[sizes[l]];
                weights[l] = new double[sizes[l]][sizes[l - 1]];
                for (int i = 0; i < sizes[l]; i++) {
                    biases[l][i] = gaussian(0.0, 0.01);
                    for (int j = 0; j < sizes[l - 1]; j++) {
                        // Heuristic: small random init scaled by sqrt(prevLayerSize)
                        weights[l][i][j] = gaussian(0.0, 1.0 / Math.sqrt(sizes[l - 1]));
                    }
                }
            }
        }

        private double gaussian(double mu, double sigma) {
            // Box-Muller
            double u = rnd.nextDouble();
            double v = rnd.nextDouble();
            double z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2 * Math.PI * v);
            return mu + sigma * z;
        }

        private double sigmoid(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        private double sigmoidPrimeFromSigmoid(double s) {
            return s * (1 - s);
        }

        // Forward pass: returns activations per layer
        public double[][] forward(double[] input) {
            int L = sizes.length;
            double[][] a = new double[L][];
            a[0] = input.clone();
            for (int l = 1; l < L; l++) {
                a[l] = new double[sizes[l]];
                for (int i = 0; i < sizes[l]; i++) {
                    double z = biases[l][i];
                    double[] w = weights[l][i];
                    double[] prev = a[l - 1];
                    for (int j = 0; j < prev.length; j++)
                        z += w[j] * prev[j];
                    a[l][i] = sigmoid(z);
                }
            }
            return a;
        }

        // Single-sample SGD update
        public void trainSample(double[] input, double[] target, double eta) {
            int L = sizes.length;
            // forward
            double[][] a = forward(input);

            // backprop deltas
            double[][] delta = new double[L][];
            int l = L - 1;
            delta[l] = new double[sizes[l]];
            for (int i = 0; i < sizes[l]; i++) {
                double ai = a[l][i];
                delta[l][i] = (ai - target[i]) * sigmoidPrimeFromSigmoid(ai);
            }

            for (l = L - 2; l >= 1; l--) {
                delta[l] = new double[sizes[l]];
                for (int i = 0; i < sizes[l]; i++) {
                    double sum = 0.0;
                    for (int k = 0; k < sizes[l + 1]; k++)
                        sum += weights[l + 1][k][i] * delta[l + 1][k];
                    delta[l][i] = sum * sigmoidPrimeFromSigmoid(a[l][i]);
                }
            }

            // update weights and biases
            for (l = 1; l < L; l++) {
                for (int i = 0; i < sizes[l]; i++) {
                    biases[l][i] -= eta * delta[l][i];
                    for (int j = 0; j < sizes[l - 1]; j++) {
                        weights[l][i][j] -= eta * delta[l][i] * a[l - 1][j];
                    }
                }
            }
        }

        public static double sampleMSE(double[] out, double[] target) {
            double s = 0.0;
            for (int i = 0; i < out.length; i++) {
                double d = out[i] - target[i];
                s += d * d;
            }
            return s / out.length;
        }

        public int predict(double[] input) {
            double[][] a = forward(input);
            double[] out = a[a.length - 1];
            int best = 0;
            for (int i = 1; i < out.length; i++)
                if (out[i] > out[best])
                    best = i;
            return best;
        }
    }

    // -----------------------
    // Helpers
    // -----------------------
    static double[] imageToInput(byte[] imageBytes) {
        double[] input = new double[imageBytes.length];
        for (int i = 0; i < imageBytes.length; i++) {
            int v = imageBytes[i] & 0xFF;
            input[i] = v / 255.0;
        }
        return input;
    }

    static double[] labelToOneHot(int label) {
        double[] v = new double[10];
        v[label] = 1.0;
        return v;
    }

    static void shuffleArray(int[] arr, Random rnd) {
        for (int i = arr.length - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = arr[i];
            arr[i] = arr[j];
            arr[j] = tmp;
        }
    }

    // -----------------------
    // Main: load, train, log MSE
    // -----------------------
    public static void main(String[] args) {
        if (args.length < 7) {
            System.out.println(
                    "Uso: java MNISTTrainer <train-images.gz> <train-labels.gz> <test-images.gz> <test-labels.gz> <epochs> <learningRate> <output-log-file> [maxTrainSamples]");
            return;
        }
        try {
            String trainImgs = args[0];
            String trainLbls = args[1];
            String testImgs = args[2];
            String testLbls = args[3];
            int epochs = Integer.parseInt(args[4]);
            double lr = Double.parseDouble(args[5]);
            String outLogFile = args[6];
            int maxTrain = Integer.MAX_VALUE;
            if (args.length >= 8)
                maxTrain = Integer.parseInt(args[7]);

            System.out.println("Carregando dataset MNIST (pode demorar)...");
            MnistDataloader trainLoader = new MnistDataloader(trainImgs, trainLbls);
            MnistDataloader.DataSet trainSet = trainLoader.load();
            MnistDataloader testLoader = new MnistDataloader(testImgs, testLbls);
            MnistDataloader.DataSet testSet = testLoader.load();

            int N = Math.min(trainSet.images.length, maxTrain);
            System.out.println("Imagens de treino carregadas: " + trainSet.images.length + " (usando N=" + N + ")");
            System.out.println("Imagens de teste carregadas: " + testSet.images.length);

            // network architecture: input -> 4 hidden layers -> output
            int inputSize = trainSet.rows * trainSet.cols;
            int[] sizes = new int[] { inputSize, 128, 64, 64, 32, 10 }; // 4 hidden layers: 128,64,64,32
            NeuralNetwork net = new NeuralNetwork(sizes, 123456L);

            // prepare indices for shuffle
            int[] indices = new int[N];
            for (int i = 0; i < N; i++)
                indices[i] = i;
            Random rnd = new Random(42);

            try (BufferedWriter log = new BufferedWriter(new FileWriter(outLogFile))) {
                log.write("Epoch, MSE\n");

                for (int e = 1; e <= epochs; e++) {
                    // shuffle
                    shuffleArray(indices, rnd);
                    double mseSum = 0.0;

                    for (int t = 0; t < N; t++) {
                        int idx = indices[t];
                        double[] input = imageToInput(trainSet.images[idx]);
                        double[] target = labelToOneHot(trainSet.labels[idx]);
                        double[][] a = net.forward(input);
                        mseSum += NeuralNetwork.sampleMSE(a[a.length - 1], target);
                        net.trainSample(input, target, lr);
                    }

                    double epochMSE = mseSum / N;
                    String line = String.format("Epoch %d: MSE = %.8f", e, epochMSE);
                    System.out.println(line);
                    log.write(String.format("%d, %.8f\n", e, epochMSE));
                    log.flush();
                }
                System.out.println("Treinamento finalizado. Log gravado em: " + outLogFile);
            }

            // pequena avaliação final (acurácia no testSet usando predict)
            int testN = testSet.images.length;
            int correct = 0;
            int evaluateN = Math.min(testN, 10000); // avaliar até 10k
            for (int i = 0; i < evaluateN; i++) {
                double[] input = imageToInput(testSet.images[i]);
                int pred = net.predict(input);
                if (pred == testSet.labels[i])
                    correct++;
            }
            double acc = 100.0 * correct / evaluateN;
            System.out.printf("Acurácia (amostra %d) = %.2f%%\n", evaluateN, acc);

        } catch (Exception ex) {
            System.err.println("Erro: " + ex.getMessage());
            ex.printStackTrace();
        }
    }
}
