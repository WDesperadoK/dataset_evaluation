package at.unisalzburg.dbresearch.apted.custom;

import at.unisalzburg.dbresearch.apted.distance.APTED;
import at.unisalzburg.dbresearch.apted.node.Node;
import at.unisalzburg.dbresearch.apted.node.StringNodeData;
import at.unisalzburg.dbresearch.apted.costmodel.StringUnitCostModel;

public class TED_with_CoreNLP_Main {
    public static void main(String[] args) {
        // Example sentences
        String sentence1 = "This is the first sentence.";
        String sentence2 = "This is another sentence.";

        // Initialize the CoreNLP-based parser
        TED_with_CoreNLPParser parser = new TED_with_CoreNLPParser();

        // Parse sentences into APTED nodes
        Node<StringNodeData> t1 = parser.fromString(sentence1);
        Node<StringNodeData> t2 = parser.fromString(sentence2);

        // Use the original StringUnitCostModel
        APTED<StringUnitCostModel, StringNodeData> apted = new APTED<>(new StringUnitCostModel());
        float ted = apted.computeEditDistance(t1, t2);

        System.out.println("Tree Edit Distance: " + ted);
    }
}
