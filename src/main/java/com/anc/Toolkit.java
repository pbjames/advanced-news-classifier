package com.anc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

public class Toolkit {
    public static List<String> listVocabulary = null;
    public static List<double[]> listVectors = null;
    private static final String FILENAME_GLOVE = "glove.6B.50d_Reduced.csv";

    public static final String[] STOPWORDS = {"a", "able", "about", "across", "after", "all", "almost", "also", "am", "among", "an", "and", "any", "are", "as", "at", "be", "because", "been", "but", "by", "can", "cannot", "could", "dear", "did", "do", "does", "either", "else", "ever", "every", "for", "from", "get", "got", "had", "has", "have", "he", "her", "hers", "him", "his", "how", "however", "i", "if", "in", "into", "is", "it", "its", "just", "least", "let", "like", "likely", "may", "me", "might", "most", "must", "my", "neither", "no", "nor", "not", "of", "off", "often", "on", "only", "or", "other", "our", "own", "rather", "said", "say", "says", "she", "should", "since", "so", "some", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "tis", "to", "too", "twas", "us", "wants", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "yet", "you", "your"};

    public File[] fileBubbleSort(File[] fArray) {
        boolean notDone = true;
        File temp;

        while (notDone) {
            notDone = false;
            for (int i = 0; i < fArray.length-1; i++) {
                String filename = fArray[i].getName();
                String filenameAhead = fArray[i+1].getName();
                if (filename.compareTo(filenameAhead) > 0) {
                    temp = fArray[i];
                    fArray[i] = fArray[i+1];
                    fArray[i+1] = temp;
                    notDone = true;
                }
            }
        }
        return fArray;
    }

    public void loadGlove() throws IOException {
        BufferedReader myReader = null;
        //TODO Task 4.1 - 5 marks
        try {
            File fp = getFileFromResource(FILENAME_GLOVE);
            FileReader fpReader = new FileReader(fp);
            myReader = new BufferedReader(fpReader);

            listVectors = new ArrayList<double[]>();
            listVocabulary = new ArrayList<String>();

            String line = myReader.readLine();
            while(line != null) {
                List<String> csLine = new ArrayList<String>(List.of(line.split(",")));
                String word = csLine.get(0);
                double[] vector = new double[50];

                for (int i = 1; i < csLine.size(); i++) {
                    vector[i-1] = Double.parseDouble(csLine.get(i));}

                listVocabulary.add(word);
                listVectors.add(vector);
                line = myReader.readLine();
            }
        } catch(Exception e){
            throw new IOException(e.getMessage());
        }
    }

    private static File getFileFromResource(String fileName) throws URISyntaxException {
        ClassLoader classLoader = Toolkit.class.getClassLoader();
        URL resource = classLoader.getResource(fileName);
        if (resource == null) {
            throw new IllegalArgumentException(fileName);
        } else {
            return new File(resource.toURI());
        }
    }

    public List<NewsArticles> loadNews() {
        List<NewsArticles> listNews = new ArrayList<>();
        //TODO Task 4.2 - 5 Marks
        try {
            File newsDir = getFileFromResource("News");
            File[] newsFiles = newsDir.listFiles();
            newsFiles = fileBubbleSort(newsFiles);

            for (File fp : newsFiles) {
                if (!(fp.getName().endsWith(".htm"))) continue;
                FileReader reader = new FileReader(fp);
                BufferedReader bReader = new BufferedReader(reader);
                StringBuilder sbHtmlCode = new StringBuilder();

                String currentLine = bReader.readLine();
                while (currentLine != null) {
                    sbHtmlCode.append(currentLine);
                    currentLine = bReader.readLine();
                }

                String htmlCode = sbHtmlCode.toString();
                String newsTitle = HtmlParser.getNewsTitle(sbHtmlCode.toString());
                String newsContent = HtmlParser.getNewsContent(sbHtmlCode.toString());
                String newsLabel = HtmlParser.getLabel(sbHtmlCode.toString());
                NewsArticles.DataType datatype = HtmlParser.getDataType(sbHtmlCode.toString());

                listNews.add(new NewsArticles(newsTitle, newsContent, datatype, newsLabel));
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

        return listNews;
    }

    public static List<String> getListVocabulary() {
        return listVocabulary;
    }

    public static List<double[]> getlistVectors() {
        return listVectors;
    }
}
