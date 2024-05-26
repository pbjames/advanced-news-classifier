package com.anc;

public class HtmlParser {
    /***
     * Extract the title of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the title if it's been found. Otherwise, return "Title not found!".
     */
    public static String getNewsTitle(String _htmlCode) {
        String titleTagOpen = "<title>";
        String titleTagClose = "</title>";

        int titleStart = _htmlCode.indexOf(titleTagOpen) + titleTagOpen.length();
        int titleEnd = _htmlCode.indexOf(titleTagClose);

        if (titleStart != -1 && titleEnd != -1 && titleEnd > titleStart) {
            String strFullTitle = _htmlCode.substring(titleStart, titleEnd);
            return strFullTitle.substring(0, strFullTitle.indexOf(" |"));
        }

        return "Title not found!";
    }

    /***
     * Extract the content of the news from the _htmlCode.
     * @param _htmlCode Contains the full HTML string from a specific news. E.g. 01.htm.
     * @return Return the content if it's been found. Otherwise, return "Content not found!".
     */
    public static String getNewsContent(String _htmlCode) {
        String contentTagOpen = "\"articleBody\": \"";
        String contentTagClose = " \",\"mainEntityOfPage\":";

        int contentStart = _htmlCode.indexOf(contentTagOpen) + contentTagOpen.length();
        int contentEnd = _htmlCode.indexOf(contentTagClose);

        if (contentStart != -1 && contentEnd != -1 && contentEnd > contentStart) {
            return _htmlCode.substring(contentStart, contentEnd).toLowerCase();
        }

        return "Content not found!";
    }

    public static NewsArticles.DataType getDataType(String _htmlCode) {
        //TODO Task 3.1 - 1.5 Marks
        String dtTag1 = "<datatype>";
        String dtTag2 = "</datatype>";

        int start = _htmlCode.indexOf(dtTag1) + dtTag1.length();
        int end = _htmlCode.indexOf(dtTag2);

        if (_htmlCode.contains(dtTag1) && end != -1 && end > start) {
            String dataType = _htmlCode.substring(start, end);

            if (dataType.equalsIgnoreCase("training")) {
                return NewsArticles.DataType.Training;
            }
        }

        return NewsArticles.DataType.Testing; //Please modify the return value.
    }

    public static String getLabel (String _htmlCode) {
        //TODO Task 3.2 - 1.5 Marks
        String labelTag1 = "<label>";
        String labelTag2 = "</label>";

        int start = _htmlCode.indexOf(labelTag1) + labelTag1.length();
        int end = _htmlCode.indexOf(labelTag2);

        if (_htmlCode.contains(labelTag1) && end != -1 && end > start) {
            return _htmlCode.substring(start, end);
        }

        return "-1"; //Please modify the return value.
    }


}