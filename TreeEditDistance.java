package at.unisalzburg.dbresearch.apted.custom;
import at.unisalzburg.dbresearch.apted.distance.APTED;
import at.unisalzburg.dbresearch.apted.costmodel.StringUnitCostModel;
import at.unisalzburg.dbresearch.apted.node.Node;
import at.unisalzburg.dbresearch.apted.node.StringNodeData;
import at.unisalzburg.dbresearch.apted.parser.BracketStringInputParser;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

import java.util.List;
import java.util.Properties;

public class TreeEditDistance {

    private static StanfordCoreNLP pipeline;

    static {
        // Initialize CoreNLP pipeline
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        pipeline = new StanfordCoreNLP(props);
    }

    // Method to parse a sentence into a bracketed tree string
    public static String parseSentence(String sentence) {
        Annotation annotation = new Annotation(sentence);
        pipeline.annotate(annotation);

        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        Tree parseTree = sentences.get(0).get(TreeCoreAnnotations.TreeAnnotation.class);
        return parseTree.toString();
    }

    // Helper method to preprocess CoreNLP parse trees for APTED
    private static String simplifyCoreNLPParse(String parseTree) {
        // Step 1: Remove extra spaces
        parseTree = parseTree.replaceAll("\\s+", "");

        // Step 2: Split node labels and terminals (e.g., DTThis -> (DT This))
        parseTree = parseTree.replaceAll("\\(([^()]+?)([A-Za-z0-9]+)\\)", "($1 $2)");

        // Step 3: Replace ".." with a single "."
        parseTree = parseTree.replace("..", ".");

        // Step 4: Ensure proper wrapping of brackets
        return parseTree;
    }

    public static void main(String[] args) {
        // Example sentences
        String sentence1 = "This is the first sentence.";
        String sentence2 = "This is another sentence.";

        // Parse sentences
        String tree1 = parseSentence(sentence1);
        String tree2 = parseSentence(sentence2);

        // Log original trees
        System.out.println("Original Tree 1: " + tree1);
        System.out.println("Original Tree 2: " + tree2);

        // Simplify trees for APTED
        String refinedTree1 = simplifyCoreNLPParse(tree1);
        String refinedTree2 = simplifyCoreNLPParse(tree2);

        // Log refined trees
        System.out.println("Refined Tree 1: " + refinedTree1);
        System.out.println("Refined Tree 2: " + refinedTree2);

        try {
            // Parse bracketed trees into APTED nodes
            BracketStringInputParser parser = new BracketStringInputParser();
            Node<StringNodeData> t1 = parser.fromString(refinedTree1);
            Node<StringNodeData> t2 = parser.fromString(refinedTree2);

            // Compute tree edit distance
            APTED<StringUnitCostModel, StringNodeData> apted = new APTED<>(new StringUnitCostModel());
            float ted = apted.computeEditDistance(t1, t2);

            System.out.println("Tree Edit Distance: " + ted);
        } catch (Exception e) {
            System.err.println("Error processing trees: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
