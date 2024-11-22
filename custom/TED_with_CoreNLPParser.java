package at.unisalzburg.dbresearch.apted.custom;

import java.io.FileWriter;
import java.io.IOException;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreAnnotations;
import at.unisalzburg.dbresearch.apted.node.Node;
import at.unisalzburg.dbresearch.apted.node.StringNodeData;

import java.util.List;
import java.util.Properties;

public class TED_with_CoreNLPParser {

    private final StanfordCoreNLP pipeline;

    public TED_with_CoreNLPParser() {
        // Initialize the CoreNLP pipeline with required annotators
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        this.pipeline = new StanfordCoreNLP(props);
    }

    public Tree parse(String sentence) {
        // Annotate the input sentence
        Annotation annotation = new Annotation(sentence);
        pipeline.annotate(annotation);

        // Get the parse tree
        List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
        return sentences.get(0).get(TreeCoreAnnotations.TreeAnnotation.class);
    }

    public Node<StringNodeData> fromString(String sentence) {
        // Parse the sentence into a CoreNLP Tree
        Tree rawTree = parse(sentence);
        // Save the result to a file
        // String outputFilePath = "parsed_trees.txt"; // Specify your file path
        // try (FileWriter writer = new FileWriter(outputFilePath, true)) { // Open in append mode
        //     String output = "Raw Parsed Tree of CoreNLP for sentence: " + rawTree.toString() + "\n";
            
        //     // Write to file
        //     writer.write(output);

        //     // Print to console for immediate feedback
        //     System.out.println(output);
        // } catch (IOException e) {
        //     System.err.println("Error writing to file: " + e.getMessage());
        // }
        // Convert the CoreNLP Tree into an APTED Node
        return convertTreeToNode(rawTree);
    }

    public Node<StringNodeData> convertTreeToNode(Tree coreNLPSubtree) {
        // Convert CoreNLP Tree to APTED-compatible Node
        Node<StringNodeData> node = new Node<>(new StringNodeData(coreNLPSubtree.value()));
        for (Tree child : coreNLPSubtree.children()) {
            node.addChild(convertTreeToNode(child));
        }
        return node;
    }

    public static Node<StringNodeData> trimTree(Node<StringNodeData> root, int layers) {
        // Base case: if the layers are 0, return the current node (leaf node)
        if (layers == 0) {
            return new Node<>(root.getNodeData());  // return a copy of the leaf node
        }

        // Create a new node to preserve the structure, for non-leaf nodes
        Node<StringNodeData> trimmedNode = new Node<>(root.getNodeData());

        // Recursively trim children, only going down to the next level of depth
        if (layers > 0) {
            for (Node<StringNodeData> child : root.getChildren()) {
                // Only trim down the children, not the leaf nodes
                Node<StringNodeData> trimmedChild = trimTree(child, layers - 1);
                trimmedNode.addChild(trimmedChild);
            }
        }
        // Return the newly trimmed node
        return trimmedNode;
    }


}
