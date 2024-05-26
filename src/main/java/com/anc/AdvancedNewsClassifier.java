package com.anc;

import org.apache.commons.lang3.time.StopWatch;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.WorkspaceMode;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class AdvancedNewsClassifier {
    public Toolkit myTK = null;
    public static List<NewsArticles> listNews = null;
    public static List<Glove> listGlove = null;
    public List<ArticlesEmbedding> listEmbedding = null;
    public MultiLayerNetwork myNeuralNetwork = null;

    public final int BATCHSIZE = 10;

    public int embeddingSize = 0;
    private static StopWatch mySW = new StopWatch();

    public AdvancedNewsClassifier() throws IOException {
        myTK = new Toolkit();
        myTK.loadGlove();
        listNews = myTK.loadNews();
        listGlove = createGloveList();
        listEmbedding = loadData();
    }

    public static void main(String[] args) throws Exception {
        mySW.start();
        AdvancedNewsClassifier myANC = new AdvancedNewsClassifier();

        myANC.embeddingSize = myANC.calculateEmbeddingSize(myANC.listEmbedding);
        myANC.populateEmbedding();
        myANC.myNeuralNetwork = myANC.buildNeuralNetwork(2);
        myANC.predictResult(myANC.listEmbedding);
        myANC.printResults();
        mySW.stop();
        System.out.println("Total elapsed time: " + mySW.getTime());
    }

    public List<Glove> createGloveList() {
        List<Glove> listResult = new ArrayList<>();
        //TODO Task 6.1 - 5 Marks

        for (int i = 0; i < Toolkit.listVocabulary.size(); i++) {
            String word = Toolkit.listVocabulary.get(i);
            Vector vector = new Vector(Toolkit.listVectors.get(i));
            boolean wordIsStop = false;

            for (String stopWord : Toolkit.STOPWORDS) if (stopWord.equals(word)) { wordIsStop = true; break; };
            if (!(wordIsStop)) listResult.add(new Glove(word, vector));
        }

        return listResult;
    }


    public static List<ArticlesEmbedding> loadData() {
        List<ArticlesEmbedding> listEmbedding = new ArrayList<>();
        for (NewsArticles news : listNews) {
            ArticlesEmbedding myAE = new ArticlesEmbedding(news.getNewsTitle(), news.getNewsContent(), news.getNewsType(), news.getNewsLabel());
            listEmbedding.add(myAE);
        }
        return listEmbedding;
    }

    public void genericBubbleSort(int[] iArray) {
        boolean notDone = true;
        int temp = 0;
        while (notDone) {
            notDone = false;
            for (int i = 0; i < iArray.length-1; i++) {
                if (iArray[i] > iArray[i+1]) {
                    temp = iArray[i];
                    iArray[i] = iArray[i+1];
                    iArray[i+1] = temp;
                    notDone = true;
                }
            }
        }
    }

    public int calculateEmbeddingSize(List<ArticlesEmbedding> _listEmbedding) {
        int intMedian = -1;
        //TODO Task 6.2 - 5 Marks
        int noEmbeddings = _listEmbedding.size();
        int[] embeddingLengths = new int[noEmbeddings];

        for (int i = 0; i < noEmbeddings; i++) {
            for (String word : _listEmbedding.get(i).getNewsContent().split(" ")) {
                if (Toolkit.getListVocabulary().contains(word)){ //&& Arrays.stream(Toolkit.STOPWORDS).noneMatch(w -> w.equals(word))) {
                    embeddingLengths[i] += 1;
                }
            }
        }

        genericBubbleSort(embeddingLengths);

        intMedian = (noEmbeddings % 2 == 0) ? (embeddingLengths[noEmbeddings/2] + embeddingLengths[(noEmbeddings/2)+1])/2 : embeddingLengths[(noEmbeddings+1)/2];

        return intMedian;
    }

    public void populateEmbedding() {
        //TODO Task 6.3 - 10 Marks
        for (ArticlesEmbedding embedding : listEmbedding) {
            try {
                embedding.getEmbedding();
            } catch (InvalidSizeException e) {
                embedding.setEmbeddingSize(calculateEmbeddingSize(listEmbedding));
            } catch (InvalidTextException e) {
                embedding.getNewsContent();
            } catch (Exception e) {
                throw new RuntimeException(e);
            } finally {
                try {
                    embedding.getEmbedding();
                } catch (Exception e) {
                    e.getMessage();
                }
            }
        }
    }

    public DataSetIterator populateRecordReaders(int _numberOfClasses) throws Exception {
        ListDataSetIterator myDataIterator = null;
        List<DataSet> listDS = new ArrayList<>();
        INDArray inputNDArray = null;
        INDArray outputNDArray = null;

        //TODO Task 6.4 - 8 Marks
        int[] shape = new int[] {1, _numberOfClasses};
        for (int i = 0; i < listNews.size(); i++) {
            if (listNews.get(i).getNewsType().equals(NewsArticles.DataType.Testing)) continue;

            inputNDArray = listEmbedding.get(i).getEmbedding();
            outputNDArray = Nd4j.create(shape);

            if (listNews.get(i).getNewsLabel().equals("1")) {
                outputNDArray = outputNDArray.put(0, 0, 1);
            } else if (listNews.get(i).getNewsLabel().equals("2")){
                outputNDArray = outputNDArray.put(0, 1, 1);
            }
            DataSet myDataSet = new DataSet(inputNDArray, outputNDArray);
            listDS.add(myDataSet);
        }

        return new ListDataSetIterator(listDS, BATCHSIZE);
    }

    public MultiLayerNetwork buildNeuralNetwork(int _numOfClasses) throws Exception {
        DataSetIterator trainIter = populateRecordReaders(_numOfClasses);
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(42)
                .trainingWorkspaceMode(WorkspaceMode.ENABLED)
                .activation(Activation.RELU)
                .weightInit(WeightInit.XAVIER)
                .updater(Adam.builder().learningRate(0.02).beta1(0.9).beta2(0.999).build())
                .l2(1e-4)
                .list()
                .layer(new DenseLayer.Builder().nIn(embeddingSize).nOut(15)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.HINGE)
                        .activation(Activation.SOFTMAX)
                        .nIn(15).nOut(_numOfClasses).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        for (int n = 0; n < 100; n++) {
            model.fit(trainIter);
            trainIter.reset();
        }
        return model;
    }

    public List<String> strBubbleSort(List<String> sArray) {
        boolean notDone = true;
        String temp;

        while (notDone) {
            notDone = false;
            for (int i = 0; i < sArray.size()-1; i++) {
                if (sArray.get(i).compareTo(sArray.get(i+1)) > 0) {
                    temp = sArray.get(i);
                    sArray.set(i, sArray.get(i+1));
                    sArray.set(i+1, temp);
                    notDone = true;
                }
            }
        }
        return sArray;
    }

    public List<String> getUniqueLabels() {
        List<String> uniqueLabels = new ArrayList<String>();
        for (ArticlesEmbedding embedding : listEmbedding) {
            String label = embedding.getNewsLabel();
            if (!(embedding.getNewsType() == NewsArticles.DataType.Testing ||
                    uniqueLabels.contains(label) || label.equals("-1"))) {
                uniqueLabels.add(label);
            }
        }
        return strBubbleSort(uniqueLabels);
    }

    public List<Integer> predictResult(List<ArticlesEmbedding> _listEmbedding) throws Exception {
        List<Integer> listResult = new ArrayList<>();
        //TODO Task 6.5 - 8 Marks
        List<String> uniqueLabels = getUniqueLabels();

        for (ArticlesEmbedding articlesEmbedding : listEmbedding) {
            if (articlesEmbedding.getNewsType() == NewsArticles.DataType.Testing) {
                INDArray embedding = articlesEmbedding.getEmbedding();
                int result = myNeuralNetwork.predict(embedding)[0];

                listResult.add(result);
                articlesEmbedding.setNewsLabel(uniqueLabels.get(result));
            }
        }

        return listResult;
    }

    public void printResults() {
        //TODO Task 6.6 - 6.5 Marks
        //this is done correctly
        List<String> uniqueLabels = getUniqueLabels();

        for (int i = 0; i < uniqueLabels.size(); i++) {
            System.out.println("Group " + (i + 1));
            for (ArticlesEmbedding embedding : listEmbedding) {
                if (embedding.getNewsType().equals(NewsArticles.DataType.Testing) &&
                    embedding.getNewsLabel().equals(uniqueLabels.get(i))) {
                    // fails because of carriage returns
                    String title = embedding.getNewsTitle();
                    System.out.println(title);
                }
            }
        }
    }
}
