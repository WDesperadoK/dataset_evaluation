package at.unisalzburg.dbresearch.apted.custom;

import at.unisalzburg.dbresearch.apted.distance.APTED;
import at.unisalzburg.dbresearch.apted.node.Node;
import at.unisalzburg.dbresearch.apted.node.StringNodeData;
import at.unisalzburg.dbresearch.apted.costmodel.StringUnitCostModel;

import com.google.gson.Gson;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class TED_with_CoreNLP_Dataset {

    // Dataset record structure
    static class Record {
        String text;
        String paraphrase;
        String domain;
        String id;
        Float ted3;  // TED-3 score
        Float tedf;  // TED-F score
    }

    public static void main(String[] args) {
        String inputFilePath = "dimple_train.json";
        String outputFilePath = "ted_results.json";

        // Initialize CoreNLP parser and APTED
        TED_with_CoreNLPParser parser = new TED_with_CoreNLPParser();
        APTED<StringUnitCostModel, StringNodeData> apted = new APTED<>(new StringUnitCostModel());

        // Read dataset
        List<Record> records = readDataset(inputFilePath);
        if (records == null) {
            System.err.println("Failed to read the dataset.");
            return;
        }

        // Variables for running average calculation
        float runningTed3 = 0;
        float runningTedf = 0;
        int validCount = 0;

        // Process records
        int processedCount = 0;
        for (Record record : records) {
            try {
                // System.out.println("Text: " + record.text);
                // System.out.println("Paraphrase: " + record.paraphrase);
                Node<StringNodeData> t1 = parser.fromString(record.text);
                Node<StringNodeData> t2 = parser.fromString(record.paraphrase);
                // System.out.println("Bracket Parsed Tree for Text: " + t1);
                // System.out.println("Bracket Parsed Tree for Paraphrase: " + t2);
                // Compute TED-3 and TED-F
                record.ted3 = computeTED3(t1, t2, apted);
                record.tedf = computeTEDF(t1, t2, apted);

                // Update running averages
                validCount++;
                runningTed3 += (record.ted3 - runningTed3) / validCount;
                runningTedf += (record.tedf - runningTedf) / validCount;

            } catch (Exception e) {
                System.err.println("Error processing Record ID: " + record.id);
                e.printStackTrace();
            }

            processedCount++;
            if (processedCount % 100 == 0) {
                System.out.println("Processed " + processedCount + " pairs...");
                System.out.println("Running Average TED-3: " + runningTed3 + ", TED-F: " + runningTedf);
            }
        }

        // Print final averages
        System.out.println("Final Average TED-3: " + runningTed3);
        System.out.println("Final Average TED-F: " + runningTedf);

        // Save results
        saveResults(outputFilePath, records);
    }

    private static Float computeTED3(Node<StringNodeData> t1, Node<StringNodeData> t2, APTED<StringUnitCostModel, StringNodeData> apted) {
        // Consider only the top 3 layers of the trees
        Node<StringNodeData> trimmedT1 = TED_with_CoreNLPParser.trimTree(t1, 3);
        Node<StringNodeData> trimmedT2 = TED_with_CoreNLPParser.trimTree(t2, 3);
        // System.out.println("Trimmed Tree for Text: " + trimmedT1);
        // System.out.println("Trimmed Tree for Paraphrase: " + trimmedT2);
        return apted.computeEditDistance(trimmedT1, trimmedT2);
    }

    private static Float computeTEDF(Node<StringNodeData> t1, Node<StringNodeData> t2, APTED<StringUnitCostModel, StringNodeData> apted) {
        // Compute the full tree edit distance
        return apted.computeEditDistance(t1, t2);
    }

    private static List<Record> readDataset(String filePath) {
        List<Record> records = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                records.add(new Gson().fromJson(line, Record.class));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return records;
    }

    private static void saveResults(String filePath, List<Record> records) {
        try (FileWriter writer = new FileWriter(filePath)) {
            new Gson().toJson(records, writer);
            System.out.println("Results saved to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
