package at.unisalzburg.dbresearch.apted.custom;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.trees.Tree;
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

    public Node<StringNodeData> convertTreeToNode(Tree coreNLPSubtree) {
        // Convert CoreNLP Tree to APTED-compatible Node
        Node<StringNodeData> node = new Node<>(new StringNodeData(coreNLPSubtree.value()));
        for (Tree child : coreNLPSubtree.children()) {
            node.addChild(convertTreeToNode(child));
        }
        return node;
    }
}
