import java.io.*;
import java.util.*;
import org.json.simple.*;
import org.json.simple.parser.*;
import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.ling.*;
import edu.stanford.nlp.trees.*;
import se.liu.ida.hefquin.aptd.*; // Import APTED library

public class EvaluateDimple {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.out.println("Usage: java EvaluateDimple <input_json> <output_log>");
            return;
        }

        String inputFile = args[0];
        String outputLog = args[1];

        // Initialize Stanford CoreNLP
        Properties props = new Properties();
        props.setProperty("annotators", "tokenize,ssplit,pos,lemma,parse");
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);

        // Read input JSON
        JSONParser parser = new JSONParser();
        JSONArray inputData = (JSONArray) parser.parse(new FileReader(inputFile));

        BufferedWriter writer = new BufferedWriter(new FileWriter(outputLog));
        writer.write("Response Pair\tTED-3\tTED-F\n");

        for (Object obj : inputData) {
            JSONObject pair = (JSONObject) obj;
            String response1 = (String) pair.get("response1");
            String response2 = (String) pair.get("response2");

            // Parse sentences into trees
            Tree tree1 = parseSentence(response1, pipeline);
            Tree tree2 = parseSentence(response2, pipeline);

            // Calculate tree edit distance
            APTED<StringNodeData> apted = new APTED<>(new StringUnitCostModel());
            int ted3 = apted.computeEditDistance(tree1, tree2);

            // Calculate normalized TED (TED-F)
            double tedf = (double) ted3 / (tree1.size() + tree2.size());

            writer.write(String.format("%s\t%d\t%.4f\n", pair.toJSONString(), ted3, tedf));
        }

        writer.close();
        System.out.println("Evaluation results saved to " + outputLog);
    }

    private static Tree parseSentence(String sentence, StanfordCoreNLP pipeline) {
        Annotation annotation = new Annotation(sentence);
        pipeline.annotate(annotation);
        return annotation.get(CoreAnnotations.SentencesAnnotation.class)
                         .get(0)
                         .get(TreeCoreAnnotations.TreeAnnotation.class);
    }
}
