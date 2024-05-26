package com.anc;

import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.pipeline.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.Properties;

public class ArticlesEmbedding extends NewsArticles {
    private int intSize = -1;
    private String processedText = "";

    private INDArray newsEmbedding = Nd4j.create(0);

    public ArticlesEmbedding(String _title, String _content, NewsArticles.DataType _type, String _label) {
        //TODO Task 5.1 - 1 Mark
        super(_title, _content, _type, _label);
    }

    public void setEmbeddingSize(int _size) {
        //TODO Task 5.2 - 0.5 Marks
        intSize = _size;
    }

    public int getEmbeddingSize(){
        return intSize;
    }

    @Override
    public String getNewsContent() {
        //TODO Task 5.3 - 10 Marks
        //this was done properly
        if (!(processedText.isEmpty())) return processedText.trim();

        String contents = super.getNewsContent();
        contents = textCleaning(contents);

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,pos,lemma");

        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        CoreDocument document = pipeline.processToCoreDocument(contents);
        StringBuilder newContents = new StringBuilder();

        for (CoreLabel tok : document.tokens()) {
            String lemma = tok.lemma();
            boolean isStopWord = false;

            for (String stopWord : Toolkit.STOPWORDS) if (stopWord.equals(lemma)) {isStopWord = true;}

            if (!(isStopWord)) newContents.append(lemma.toLowerCase()).append(" ");
        }

        processedText = newContents.toString();
        return processedText.trim();
    }

    public INDArray getEmbedding() throws Exception {
        //TODO Task 5.4 - 20 Marks
        if (intSize == -1) throw new InvalidSizeException("Invalid size");
        if (processedText.isEmpty()) throw new InvalidTextException("Invalid text");

        int[] shape = {intSize, 50};
        INDArray newsEmbedding = Nd4j.create(shape);

        String[] validWords = new String[processedText.split(" ").length];

        int count = 0;
        for (String procWord : processedText.split(" ")) {
            if (Toolkit.listVocabulary.contains(procWord)) {
                validWords[count] = procWord;
                count += 1;
            }
        }

        for (int i = 0; i < intSize; i++) {
            if (i >= validWords.length) {
                newsEmbedding.putRow(i, Nd4j.create(new int[50]));
            } else {
                int idxVocab = Toolkit.getListVocabulary().indexOf(validWords[i]);
                if (idxVocab == -1) {continue;}
                newsEmbedding.putRow(i, Nd4j.create(Toolkit.getlistVectors().get(idxVocab)));
            }
        }

        return Nd4j.vstack(newsEmbedding.mean(1));
    }

    /***
     * Clean the given (_content) text by removing all the characters that are not 'a'-'z', '0'-'9' and white space.
     * @param _content Text that need to be cleaned.
     * @return The cleaned text.
     */
    private static String textCleaning(String _content) {
        StringBuilder sbContent = new StringBuilder();

        for (char c : _content.toLowerCase().toCharArray()) {
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || Character.isWhitespace(c)) {
                sbContent.append(c);
            }
        }

        return sbContent.toString().trim();
    }
}
