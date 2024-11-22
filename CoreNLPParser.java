import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;

public class CoreNLPParser {
    public static String parseSentence(String sentence) {
        // Set up CoreNLP properties
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Annotate the sentence
        Annotation annotation = new Annotation(sentence);
        pipeline.annotate(annotation);

        // Extract the parse tree
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        Tree parseTree = sentences.get(0).get(TreeCoreAnnotations.TreeAnnotation.class);

        // Return the tree as a bracketed string
        return parseTree.toString();
    }

    public static void main(String[] args) {
        String sentence = "This is a test sentence.";
        String parseTree = parseSentence(sentence);
        System.out.println("Parse Tree: " + parseTree);
    }
}
