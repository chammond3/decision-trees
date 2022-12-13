/****
Author: Charlie Hammond
Project 3 - Decision Trees
Date: 17 October 2021
Purpose: This project implements decision trees
****/

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DecisionTrees {
    public static void main( String[] args )
	{
        int dataSetNumber = 5; // 0 = breast cancer, 1 = cars, 2 = voting, 3 = abalone, 4 = computer, 5 = forest fire, 6 = weather data lecture example
        double errorThreshold = 10000; // early stopping MSE for regression datasets (added the small value b/c of weird double comparison in java)
        boolean prune = true;
        boolean tune = false;

        //double data[][] = DataStream.getData(dataSetNumber, 0);

        boolean[] categoricalAttribute = DataStream.getCatArray(dataSetNumber);
        double[][] tuneAndPruneData = DataStream.getData(dataSetNumber, 1); // 20% used for early stopping - pruning or tuning error
        double[][] testData = DataStream.getData(dataSetNumber, 2); // 80% used 

        double[][] data;
        if (tune && !categoricalAttribute[categoricalAttribute.length - 1]) { // only tune regression data sets
            data = tuneAndPruneData;
        } else {
            data = testData;
        }

        /* // print data
        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[0].length; j++) {
                System.out.print(data[i][j] + " "); 
            }
            System.out.print("\n");
        }*/

        double[] results = new double[data.length];
        int[] kFoldArray = DataStream.getKFold(dataSetNumber, 5, data);

        // loop over each of the k folds, create a tree for remaining k-1 folds and test with k fold
        for (int i = 0; i < 5; i++) {

            // create the current training set out of k-1 folds
            int currentTrainingSetLength = 0;
            for (int j = 0; j < data.length; j++) {
                if (kFoldArray[j] != i) {
                    currentTrainingSetLength++;
                }
            }
            
            // make training set from k-1 folds
            double[][] currentTrainingSet = new double[currentTrainingSetLength][data[0].length];
            int currentIndex = 0;
            for (int j = 0; j < data.length; j++) {
                if (kFoldArray[j] != i) {
                    for (int k = 0; k < data[0].length; k++) {
                        currentTrainingSet[currentIndex][k] = data[j][k];
                    }
                    currentIndex++;
                }
            }
            // generate tree from the data
            Node decisionTree = startGenerateTree(currentTrainingSet, categoricalAttribute, errorThreshold + 0.0001);

            decisionTree.printTree("");

            // prune with the prune data set
            if (prune && categoricalAttribute[categoricalAttribute.length - 1]) {
                pruneTree(decisionTree, decisionTree, tuneAndPruneData, dataSetNumber);
                System.out.print("\n\nPruned Tree\n");
                decisionTree.printTree("");
            }
            
            // evaulate tree with the remaining 1 fold
            evaluateTree(decisionTree, data, i, kFoldArray, results);
        }
        
        // print results
        for (int i = 0; i < data.length; i++) {
            System.out.println(results[i] + " " + data[i][data[0].length - 1]);
        }

        // calculate performance (MSE or classification accuracy)
        double performance = DataStream.calculatePerformance(dataSetNumber, results, data);
        System.out.println("Performance: " + performance);
    }

    /****
	 * Method: evaluateTree
	 * Description: takes a tree and test data and makes predictions based on the attributes
	****/
    public static void evaluateTree(Node root, double[][] data, int kFold, int[] kFoldArray, double[] results) {
        for (int i = 0; i < data.length; i++) {
            if (kFold == -1 || kFoldArray[i] == kFold) {
                Node currentNode = root;
                double currentMostFrequent = currentNode.mostFrequentClass;
                while (currentNode != null && !currentNode.isLeaf && !currentNode.isPruned) {
                    // determine the feature this node splits on
                    int splitIndex = currentNode.splitIndex;

                    // is attribute categorical?
                    if (root.categoricalAttribute[splitIndex]) {
                        // find the most common class in case we can't classify the point
                        currentMostFrequent = currentNode.mostFrequentClass;

                        // traverse to child node based on data's feature value
                        currentNode = currentNode.childNodes.get(data[i][splitIndex]);
                        
                    } else { // numeric attribute - binary split
                        // take left child
                        if (data[i][splitIndex] > currentNode.splitValue) {
                            currentNode = currentNode.childNodes.get(0.0);
                        } else {
                            currentNode = currentNode.childNodes.get(1.0);
                        }
                    }
                }
                // set results to leaf node
                if (currentNode == null || currentNode.isPruned) { // this is a new combination of attributes that wasn't trained
                    results[i] = currentMostFrequent;
                } else {
                    results[i] = currentNode.leafValue;
                }
            }
        }
    }

    /****
	 * Method: startGenerateTree
	 * Description: begins generation of a decision tree
	****/
    public static Node startGenerateTree(double[][] data, boolean[] categoricalAttribute, double errorThreshold) {
        Node root = new Node(data, categoricalAttribute);
        //root.printData(data);
        generateTree(root, "", errorThreshold);
        return root;  
    }

    /****
	 * Method: generateTree
	 * Description: recursive function that determines splits on the current node
	****/
    public static void generateTree(Node root, String offset, double errorThreshold) {
        // stopping case
        boolean isClassification = root.categoricalAttribute[root.categoricalAttribute.length - 1];
        if (isClassification) {
            root.calculateEntropy();
        } else {
            root.calculateMSE();
        }
        
        if (isClassification && (root.entropy == 0 || root.noPossibleSplit())) {
            // make node into leaf and return
            root.makeLeaf();
            //System.out.println(offset + "Leaf value = " + root.leafValue);
            return; 
        } else if (!isClassification && root.meanSquaredError <= errorThreshold) {
            root.makeLeaf();
            //System.out.println(offset + "Leaf value = " + root.leafValue);
        } else { // need to split futher
            int splitIndex;
            if (isClassification) {
                splitIndex = root.determineSplitClassification();
            } else {
                splitIndex = root.determineSplitRegression();
            }
            
            // entropy is all the same from splits - will keep splitting on same attribute
            if (splitIndex == -1) {
                root.makeLeaf();
                //System.out.println(offset + "Leaf value = " + root.leafValue);
                return;
            }

            // split the node
            root.split(splitIndex);
            //System.out.println(offset + "Split on attribute " + splitIndex);

            // loop over children and generate a tree from each child
            for (Map.Entry<Double, Node> entry: root.childNodes.entrySet()) {
                /* // print as it is grown
                if (root.categoricalAttribute[splitIndex]) { // splitting on a category attribute
                    System.out.println(offset + "Split value: " + entry.getKey());
                } else {
                    if (entry.getKey() == 0.0) { // left split
                        System.out.println(offset + "Split value < " + root.splitValue);
                    } else {
                        System.out.println(offset + "Split value > " + root.splitValue);
                    }
                }*/
                generateTree(entry.getValue(), offset + "   ", errorThreshold);
            }
            
        }
    }
    
    /****
	 * Method: pruneTree
	 * Description: prunes classification trees. Recursively traverses tree and sees if removing the node helps the performance based on a pruning set
	****/
    public static void pruneTree(Node currentNode, Node root, double[][] pruneData, int dataSetNumber) {
        // stopping case if leaf
        if (currentNode.isLeaf) {
            return;
        } else {
            // recur for each child node
            for (Map.Entry<Double, Node> entry: currentNode.childNodes.entrySet()) {
                pruneTree(entry.getValue(), root, pruneData, dataSetNumber);
            }

            double[] results = new double[pruneData.length];
            int[] kFoldArray = new int[pruneData.length]; 
            // evaluate performance for existing pruned and with current node pruned
            evaluateTree(root, pruneData, -1, kFoldArray, results);
            double currentPerformance = DataStream.calculatePerformance(dataSetNumber, results, pruneData);

            // prune the node
            currentNode.isPruned = true;

            evaluateTree(root, pruneData, -1, kFoldArray, results);
            double prunedPerformance = DataStream.calculatePerformance(dataSetNumber, results, pruneData);

            // if pruning makes tree worse, revert it to not pruned
            if (prunedPerformance < currentPerformance) {
                currentNode.isPruned = false;
            } 
            return;
        }
    }
}



class Node {
    double[][] data;  // keep track of the data at the current node when building the tree
    boolean[] categoricalAttribute; // = {true, true, true, true, true};
    HashMap<Double, Node> childNodes = new HashMap<Double, Node>(); // keep track of children -> could have any number of children if split on categorical attribute
    int splitIndex; // attribute to split on
    int splitRow; // row on which to split if it splits on a numeric attribute
    double splitValue; // split condition if numeric attribute
    double entropy; // entropy value at node
    double meanSquaredError; // MSE at node - used on regression datasets
    int[][] frequency; // the frequency of each category value across the data
    boolean isLeaf; // is this node a leaf node?
    double leafValue; // value at the leaf
    double mostFrequentClass; // value of most frequent class with training subset at the node
    boolean isPruned; // flag if the current node has been removed with pruning

    /****
	 * Method: Node constructor
	 * Description: takes data and categorical attribute array
	****/
    Node(double[][] data, boolean[] categoricalAttribute) {
        this.data = data;
        this.categoricalAttribute = categoricalAttribute;
        this.calculateFrequency();
        this.isLeaf = false;
        this.isPruned = false;
    }

    /****
	 * Method: calculateEntropy
	 * Description: calculates the entropy of a node
	****/
    public void calculateEntropy() {
        int rows = data.length;
        int columns = data[0].length;
        double[] sortedY = new double[rows];

        for (int i = 0; i < rows; i++) {
            sortedY[i] = data[i][columns - 1];
        }

        Arrays.sort(sortedY);

        int currentCount = 1;
        entropy = 0;
        for (int i = 0; i < rows; i++) {
            if (i == (rows - 1) || sortedY[i] != sortedY[i + 1]) {
                entropy = entropy - ((double) currentCount / rows) * Math.log((double) currentCount / rows);
                currentCount = 1;
            } else {
                currentCount++;
            }
        }
    }

    /****
	 * Method: calculateMSE
	 * Description: calculates the MSE of a node
	****/
    public void calculateMSE() {
        double meanSquared = 0;
        double average = 0;
        for (int i = 0; i < data.length; i++) {
            average = average + (data[i][data[0].length - 1] / data.length);
        }

        for (int i = 0; i < data.length; i++) {
            meanSquared = meanSquared + ((data[i][data[0].length - 1] - average) * (data[i][data[0].length - 1] - average));
        }
        this.meanSquaredError = meanSquared / data.length;
        
    }

    /****
	 * Method: split
	 * Description: adds child nodes to current node
	****/
    public void split(int splitIndex) {
        double[][] sortedData = new double[data.length][data[0].length];
        this.splitIndex = splitIndex;

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                sortedData[i][j] = data[i][j];
            }
        }

        Arrays.sort(sortedData, (row1, row2) -> Double.compare(row1[splitIndex], row2[splitIndex]));
        
        if (categoricalAttribute[splitIndex]) {
            int currentRow = 0;
            int currentLastRow;
            for (int i = 0; i < frequency[splitIndex].length; i++) {
                currentLastRow = currentRow + frequency[splitIndex][i];
                double[][] childData = new double[currentLastRow - currentRow][data[0].length];
                int currentChildRow = 0;
                for (int j = currentRow; j < currentLastRow; j++) {
                    for (int k = 0; k < data[0].length; k++) {
                        childData[currentChildRow][k] = sortedData[j][k]; 
                    }
                    currentChildRow++;
                }
                Node childNode = new Node(childData, this.categoricalAttribute);
                childNodes.put(sortedData[currentRow][splitIndex], childNode);
                currentRow = currentLastRow;
            }
        } else {
            double[][] childDataLeft = new double[splitRow][data[0].length];
            double[][] childDataRight = new double[data.length - splitRow][data[0].length];
            for (int i = 0; i < data.length; i++) {
                if (i >= splitRow) {
                    for (int j = 0; j < data[0].length; j++) {
                        childDataRight[i - splitRow][j] = data[i][j];
                    }
                } else {
                    for (int j = 0; j < data[0].length; j++) {
                        childDataLeft[i][j] = data[i][j];
                    }
                }
            }
            Node childNodeLeft = new Node(childDataLeft, this.categoricalAttribute);
            Node childNodeRight = new Node(childDataRight, this.categoricalAttribute);
            childNodes.put(0.0, childNodeLeft);
            childNodes.put(1.0, childNodeRight);
        }
    }

    /****
	 * Method: determinesSplitRegression
	 * Description: determines which attribute to split on based on minimizing MSE
	****/
    public int determineSplitRegression() {
        double[] meanSquaredError = new double[data[0].length - 1];
        int[] splitRowArray = new int[data[0].length - 1];
        double[] splitValueArray = new double[data[0].length - 1];
        
        // deep copy of data so we can sort it
        double[][] sortedData = new double[data.length][data[0].length];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                sortedData[i][j] = data[i][j];
            }
        }

        for (int column = 0; column < meanSquaredError.length; column++) {
            final int sortColumn = column;
            //Arrays.sort(sortedData, (row1, row2) -> Double.compare(row1[data[0].length - 1], row2[data[0].length - 1]));
            Arrays.sort(sortedData, (row1, row2) -> Double.compare(row1[sortColumn], row2[sortColumn]));
            //this.printData(sortedData);

            // determine splits if attribute is categorical
            if (categoricalAttribute[column]) {
                // find average result based on each attribute value
                double[] averages = new double[frequency[column].length]; // n = number of unique attributes
                int currentBin = 0;
                for (int i = 0; i < data.length; i++) {
                    averages[currentBin] = averages[currentBin] + sortedData[i][data[0].length - 1] / frequency[column][currentBin];

                    if (i != (data.length - 1) && sortedData[i][column] != sortedData[i + 1][column]) {
                        currentBin++;
                    }
                }
                // calculate MSE based on averages
                currentBin = 0;
                for (int i = 0; i < data.length; i++) {
                    double currentMSE = (sortedData[i][data[0].length - 1] - averages[currentBin]) * (sortedData[i][data[0].length - 1] - averages[currentBin]);
                    meanSquaredError[column] = meanSquaredError[column] + currentMSE;

                    if (i != (data.length - 1) && sortedData[i][column] != sortedData[i + 1][column]) {
                        currentBin++;
                    }
                
                }
            } else { // numeric attribute
                int middleIndex = sortedData.length / 2;
                splitRowArray[column] = middleIndex;
                // set the value on which to split as the average of the two adjacent rows
                splitValueArray[column] = (sortedData[middleIndex - 1][column] + sortedData[middleIndex][column]) / 2;
                double lowAverage = 0; // left split average
                double highAverage = 0; // right split average
                for (int i = 0; i < sortedData.length; i++) {
                    if (i >= middleIndex) {
                        highAverage = highAverage + sortedData[i][sortedData[0].length - 1] / (sortedData.length - middleIndex);
                    } else {
                        lowAverage = lowAverage + sortedData[i][sortedData[0].length - 1] / middleIndex;
                    }
                }

                for (int i = 0; i < sortedData.length; i++) {
                    double currentMSE;
                    if (i >= middleIndex) {
                        currentMSE = (sortedData[i][data[0].length - 1] - highAverage) * (sortedData[i][data[0].length - 1] - highAverage);
                    } else {
                        currentMSE = (sortedData[i][data[0].length - 1] - lowAverage) * (sortedData[i][data[0].length - 1] - lowAverage);
                    }
                    meanSquaredError[column] = meanSquaredError[column] + currentMSE;
                }

            }
            meanSquaredError[column] = meanSquaredError[column] / sortedData.length;
        
        }

        // find minimum mean squared error
        int splitIndex = 0;
        double minMSE = meanSquaredError[0];
        boolean allSameMSE = true;
        for (int i = 1; i < meanSquaredError.length; i++) {
            if (meanSquaredError[i] < minMSE) {
                splitIndex = i;
                minMSE = meanSquaredError[i];
            }

            // determine if all of the splits are identical - this will result in infinite loop
            double currentMSE = Math.round(meanSquaredError[i] * 10000); // round off a few decimal places - I ws getting weird comparision issues that I think relates to how java stores doubles
            double currentMinMSE = Math.round(minMSE * 10000);
            if (allSameMSE && (currentMSE != currentMinMSE)) {
                allSameMSE = false;
            }
        }

        // determine the row we should split on and the comparison value for moving forward
        if (!categoricalAttribute[splitIndex]) {
            this.splitRow = splitRowArray[splitIndex];
            this.splitValue = splitValueArray[splitIndex];
        }

        if (allSameMSE) {
            splitIndex = -1;
        }
        
        return splitIndex;
    }

    /****
	 * Method: determineSplitClassification
	 * Description: determines the split attribute for classification data sets. Maximizes the gain ratio
	****/
    public int determineSplitClassification() {
        double[] featureEntropy = new double[data[0].length - 1];
        double[] gainRatio = new double[data[0].length - 1];

        // deep copy of data so we can sort it
        double[][] sortedData = new double[data.length][data[0].length];

        for (int i = 0; i < data.length; i++) {
            for (int j = 0; j < data[i].length; j++) {
                sortedData[i][j] = data[i][j];
            }
        }

        for (int column = 0; column < gainRatio.length; column++) {
            int categoryIndex = 0; // index in frequency array
            double currentEntropy = 0;
            int currentCount = 1;

            final int sortColumn = column;
            Arrays.sort(sortedData, (row1, row2) -> Double.compare(row1[data[0].length - 1], row2[data[0].length - 1]));
            Arrays.sort(sortedData, (row1, row2) -> Double.compare(row1[sortColumn], row2[sortColumn]));

            for (int i = 0; i < (data.length - 1); i++) {
                // if not the same category - go to next one
                if (sortedData[i][column] != sortedData[i + 1][column]) {
                    double currentFrequency = (double) currentCount / this.frequency[column][categoryIndex];
                    currentEntropy = currentEntropy - (currentFrequency * Math.log(currentFrequency));
                    featureEntropy[column] = featureEntropy[column] + (((double) this.frequency[column][categoryIndex] / data.length)* currentEntropy);
                    currentEntropy = 0;
                    currentCount = 1;
                    categoryIndex++;
                } else { // same category - calculate entropy for split
                    // if the class of the current row is not equal to the next 
                    if (sortedData[i][data[0].length - 1] != sortedData[i + 1][data[0].length - 1]) {
                        double currentFrequency = (double) currentCount / this.frequency[column][categoryIndex];
                        currentEntropy = currentEntropy - (currentFrequency * Math.log(currentFrequency)); 
                        currentCount = 1;
                    } else {
                        currentCount++;
                    }
                }

                // clean up if last iteration
                if (i == data.length - 2) {
                    double currentFrequency = (double) currentCount / this.frequency[column][categoryIndex];
                    currentEntropy = currentEntropy - (currentFrequency * Math.log(currentFrequency));
                    featureEntropy[column] = featureEntropy[column] + (((double) this.frequency[column][categoryIndex] / data.length)* currentEntropy);
                } 
            }
            // calculate information value
            double currentInformationValue = 0;
            for (int i = 0; i < frequency[column].length; i++) {
                currentInformationValue = currentInformationValue - (((double) frequency[column][i] / data.length) * Math.log((double) frequency[column][i] / data.length));
            }
            // if all same value, no information  gain
            if (currentInformationValue == 0) {
                gainRatio[column] = 0;
            } else {
                gainRatio[column] = (this.entropy - featureEntropy[column]) / currentInformationValue;
            }
            
        }

        int splitIndex = 0;
        double minEntropy = featureEntropy[0];
        boolean allSameEntropy = true;
        double maxGainRatio = gainRatio[0];
        for (int i = 1; i < featureEntropy.length; i++) {
            // find max gain ratio
            if (gainRatio[i] > maxGainRatio) {
                splitIndex = i;
                maxGainRatio = gainRatio[i];
            }
    
            // determine if all of the splits are identical - this will result in infinite loop
            if (featureEntropy[i] < minEntropy) {
                
                minEntropy = featureEntropy[i];
            }

            if (allSameEntropy && featureEntropy[i] != minEntropy) {
                allSameEntropy = false;
            }
        }

        if (allSameEntropy) {
            splitIndex = -1;
        }

        return splitIndex;
    }

    /****
	 * Method: makeLeaf
	 * Description: makes a node into a leaf when we won't split anymore
	****/
    public void makeLeaf() {
        this.isLeaf = true;

        // determine leaf value
        int rows = data.length;
        int columns = data[0].length;
        double[] sortedY = new double[rows];

        for (int i = 0; i < rows; i++) {
            sortedY[i] = data[i][columns - 1];
        }

        Arrays.sort(sortedY);

        if (categoricalAttribute[categoricalAttribute.length - 1]) {
            // find the majority category
            int maxCount = 1;
            double maxValue = sortedY[0];
            int currentCount = 1;
            for (int i = 0; i < (rows - 1); i++) {
                if (sortedY[i] == sortedY[i + 1]) {
                    currentCount++;
                    if (currentCount > maxCount) {
                        maxCount = currentCount;
                        maxValue = sortedY[i];
                    }
                } else {
                    currentCount = 1;
                }
            }

            this.leafValue = maxValue;
        } else { //leaf = average for regression
            double sum = 0;
            for (int i = 0; i < rows; i++) {
                sum = sum + data[i][data[0].length - 1];
            }
            this.leafValue = sum / rows;
        }

    }

    /****
	 * Method: noPossibleSplit
	 * Description: used to determine if there is no split that reduces entropy
	****/
    public boolean noPossibleSplit() {
        boolean noPossibleSplit = true;
        int row = 0;
        int rows = data.length;
        int columns = data[0].length;

        while (row < (rows - 1) && noPossibleSplit) {
            int column = 0;
            while (column < (columns - 1) && noPossibleSplit) {
                if (data[row][column] != data[row + 1][column]) {
                    noPossibleSplit = false;
                }
                column++;
            }
            row++;
        }

        return noPossibleSplit;
    }

    /****
	 * Method: calculateFrequency
	 * Description: Populates the frequency array based on the frequency of each value for an attribute. Used to help calculate entropy
	****/
    private void calculateFrequency() {
        frequency = new int[data[0].length][];
        for (int column = 0; column < data[0].length; column++) {
            if (categoricalAttribute[column]) {
                double[] sortedData = new double[data.length];
                for (int i = 0; i < data.length; i++) {
                    sortedData[i] = data[i][column];
                }
                Arrays.sort(sortedData);

                int uniqueCount = 1;
                for (int i = 0; i < (sortedData.length - 1); i++) {
                    if (sortedData[i] != sortedData[i + 1]) {
                        uniqueCount++;
                    }
                }
                frequency[column] = new int[uniqueCount];

                // class label - save off most frequent class label - used if there is no matching child node when testing data
                if (column == (data[0].length - 1)) {
                    int maxCount = 1;
                    double maxValue = sortedData[0];
                    int currentCount = 1;
                    for (int i = 0; i < (sortedData.length - 1); i++) {
                        if (sortedData[i] != sortedData[i + 1]) {
                            currentCount = 1;
                        } else {
                            currentCount++;
                            if (currentCount > maxCount) {
                                maxCount = currentCount;
                                maxValue = sortedData[i];
                            }
                        }
                    }
                    this.mostFrequentClass = maxValue;
                }

                // determine the frequency of each value
                int currentValueCount = 1;
                int currentIndex = 0;
                for (int i = 0; i < (sortedData.length - 1); i++) {
                    if (sortedData[i] != sortedData[i + 1]) {
                        frequency[column][currentIndex] = currentValueCount;
                        currentIndex++;
                        currentValueCount = 1;
                    } else {
                        currentValueCount++;
                    } 
                    if (i == (sortedData.length - 2)) {
                        frequency[column][currentIndex] = currentValueCount;
                    }
                }
            }
        }
    }

    public void printTree(String offset) {
        if (this.isLeaf) {
            System.out.println(offset + "Leaf value = " + this.leafValue);
        } else if (this.isPruned) {
            System.out.println(offset + "Leaf value = " + this.mostFrequentClass);
        } else {
            System.out.println(offset + "Split on attribute " + splitIndex);

            // loop over children and generate a tree from each child
            for (Map.Entry<Double, Node> entry: this.childNodes.entrySet()) {
                if (this.categoricalAttribute[splitIndex]) { // splitting on a category attribute
                    System.out.println(offset + "Split value: " + entry.getKey());
                } else {
                    if (entry.getKey() == 0.0) { // left split
                        System.out.println(offset + "Split value < " + this.splitValue);
                    } else {
                        System.out.println(offset + "Split value > " + this.splitValue);
                    }
                }
                entry.getValue().printTree(offset + "   ");
            }
        }
    }

    /****
	 * Method: printData
	 * Description: prints an array
	****/
    public void printData(double[][] newData) {
        System.out.print("\n");
        for (int i = 0; i < newData.length; i++) {
            for (int j = 0; j < newData[i].length; j++) {
                System.out.print(newData[i][j] + " ");
            }
            System.out.print("\n");
        }
    }

    }