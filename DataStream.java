import java.io.*;
import java.util.Arrays;

public class DataStream {
    // tuning = 0 => full set, 1 => 20%, 2 => 80%
    public static double[][] getData(int dataSetNumber, int tuning) {
        // test data example from lecture
        if (dataSetNumber == 6) {
            double[][] data = {
                {0, 0, 0, 0, 0},
                {0, 0, 0, 1, 0},
                {1, 0, 0, 0, 1},
                {2, 1, 0, 0, 1},
                {2, 2, 1, 0, 1},
                {2, 2, 1, 1, 0},
                {1, 2, 1, 1, 1},
                {0, 1, 0, 0, 0},
                {0, 2, 1, 0, 1},
                {2, 1, 1, 0, 1},
                {0, 1, 1, 1, 1},
                {1, 1, 0, 1, 1},
                {1, 0, 1, 0, 1},
                {2, 1, 0, 1, 0},
            };
            return data;
        }

        // set data size and input file based on data
		int columnCount = getColumnCount(dataSetNumber);
		int rowCount = getRowCount(dataSetNumber);
		String readFile = getReadFile(dataSetNumber);
        
        double[][] inputData = new double[rowCount][columnCount];


        // only standardize regression set
        boolean standardization = false;
        
        File readFileValid = new File(readFile);
		if (!readFileValid.canRead())
		{
			System.out.println("cannot read input");
			System.exit(1);
		}
		BufferedReader inputFile = null;
		// read data from input array, parse, and put into inputData
		try 
		{
			// read data from file
			inputFile = new BufferedReader(new FileReader(readFile));
			for (int i = 0; i < rowCount; i++) {
				String currentLine = inputFile.readLine();
				if (dataSetNumber == 5 && i == 0) { // throw away first line in fire data
					currentLine = inputFile.readLine();
				}
				double[] inputLineClean= cleanRow(dataSetNumber, currentLine);
				for (int j = 0; j <= (columnCount - 1); j++) {
					inputData[i][j] = inputLineClean[j];
				}
			}

			// perform action on the data
			fixMissingValues(dataSetNumber, inputData);

			if (standardization) {
				standardizeData(dataSetNumber, inputData);
			}

			


		}
		catch (IOException except)
		{
			System.exit(1); // don't keep going if there's a problem reading the file.
		}

		// close files
		try
		{
			inputFile.close();
		}
		catch (Exception except)
		{
			System.exit(1);
		}

        // pull out tuning data
        int tuningSize = (rowCount / 5) + 1; // 20%
        double[][] tuningData = new double[tuningSize][columnCount];
        double[][] testData = new double[rowCount - tuningSize][columnCount];


        // return full set, tuning set, or test set (based on tuning)
        if (tuning == 0) {
            return inputData;
        }
        if (tuning == 1) {
            tuningData = splitData(dataSetNumber, inputData, true);
            return tuningData;
        } else {
            testData = splitData(dataSetNumber, inputData, false);
            return testData;
        }
    }

    /****
	 * Method: getCatArray
	 * Description: returns an array of boolean values for whether an attribute is categorical or numeric
	****/
    public static boolean[] getCatArray(int dataSetNumber) {
        boolean[] categoricalAttribute;

        if (dataSetNumber == 0) { // breast cancer
            categoricalAttribute = new boolean[] {true, true, true, true, true, true, true, true, true, true};
        } else if (dataSetNumber == 1) { // car
            categoricalAttribute = new boolean[] {true, true, true, true, true, true, true};
        } else if (dataSetNumber == 2) { // house votes
            categoricalAttribute = new boolean[] {true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true, true};
        } else if (dataSetNumber == 3) { // abalone
            categoricalAttribute = new boolean[] {true, true, true, false, false, false, false, false, false, false, false};
        } else if (dataSetNumber == 4) { // machine
            categoricalAttribute = new boolean[] {false, false, false, false, false, false, false, false};
        } else if (dataSetNumber == 5) { // forest fires
            categoricalAttribute = new boolean[] {false, false, true, true, false, false, false, false, false, false, false, false, false};
        } else {
            categoricalAttribute = new boolean[] {true, true, true, true, true};
        }

        return categoricalAttribute;
    }

    /****
	 * Method: calculate performance
	 * Description: takes results and original data and determines performance of results. 
	****/
	public static double calculatePerformance(int dataSetNumber, double[] results, double[][] data) {
		boolean classifcation = getClassification(dataSetNumber);

		int rows = data.length; //getRowCount(dataSetNumber);
		int yColumn = getColumnCount(dataSetNumber) - 1;

		// classification accuracy
		if (classifcation) {
			int correctCount = 0;
			for (int i = 0; i < rows; i++) {
				if (results[i] == data[i][yColumn]) {
					correctCount++;
				}
			}
			
			double accuracy = (double) correctCount / rows;
			return accuracy;
		} else {
			// mean squared error
			double meanSquared = 0;
			for (int i = 0; i < rows; i++) {
				meanSquared = meanSquared + ((data[i][yColumn] - results[i]) * (data[i][yColumn] - results[i]));
			}
			double meanSquaredError = meanSquared / rows;
			return meanSquaredError;
		}	
	}

    /****
	 * Method: kFold
	 * Description: bins data into 5 folds to cross validation
	****/
	public static int[] getKFold(int dataSetNumber, int k, double[][] data) {
		int rows = data.length; //getRowCount(dataSetNumber);
		int yColumn = getColumnCount(dataSetNumber) - 1;
		double[][] sortedY = new double[rows][2]; // y value, original row index

		// sort y values
		for (int i = 0; i < rows; i++) {
			sortedY[i][0] = data[i][yColumn];
			sortedY[i][1] = i;
		}
		Arrays.sort(sortedY, (row1, row2) -> Double.compare(row1[0], row2[0]));
		
		int[] kFoldArray = new int[rows];
		for (int i = 0; i < rows; i++) {
			int bin = i % k;
			int rowIndex = (int) sortedY[i][1];
			kFoldArray[rowIndex] = bin;
		}
		
		return kFoldArray;
    }

	/****
	 * Method: cleanRow
	 * Description: cleans one row of input data and converts text to strings
	****/
	private static double[] cleanRow(int dataSetNumber, String inputLine) {
		String[] inputLineSplit = inputLine.split(",");
		int columnCount = getColumnCount(dataSetNumber);
		double[] cleanData = new double[columnCount];

		if (dataSetNumber == 0) { // breast cancer data
			// loop over fields - start at 1 to drop sample code number
			for (int i = 1; i < 11; i++) {
				if (inputLineSplit[i].equals("?")) {
					cleanData[i - 1] = 0; // set to zero if blank, will fix later
				}
				else {
					cleanData[i - 1] = Double.parseDouble(inputLineSplit[i]);
				}
			}
		} else if (dataSetNumber == 1) { // car data			
			// buying price
			if (inputLineSplit[0].equals("v-high")) {
				cleanData[0] = 3;
			} else if (inputLineSplit[0].equals("high")) {
				cleanData[0] = 2;
			} else if (inputLineSplit[0].equals("med")) {
				cleanData[0] = 1;
			} else if (inputLineSplit[0].equals("low")) {
				cleanData[0] = 0;
			}

			// maintenance price
			if (inputLineSplit[1].equals("v-high")) {
				cleanData[1] = 3;
			} else if (inputLineSplit[1].equals("high")) {
				cleanData[1] = 2;
			} else if (inputLineSplit[1].equals("med")) {
				cleanData[1] = 1;
			} else if (inputLineSplit[1].equals("low")) {
				cleanData[1] = 0;
			}

			// doors
			if (inputLineSplit[2].equals("5more")) {
				cleanData[2] = 5;
			} else {
				cleanData[2] = Integer.parseInt(inputLineSplit[2]);
			}

			// persons
			if (inputLineSplit[3].equals("more")) {
				cleanData[3] = 5;
			} else {
				cleanData[3] = Integer.parseInt(inputLineSplit[3]);
			}

			// size of luggage boot
			if (inputLineSplit[4].equals("big")) {
				cleanData[4] = 2;
			} else if (inputLineSplit[4].equals("med")) {
				cleanData[4] = 1;
			} else if (inputLineSplit[4].equals("small")) {
				cleanData[4] = 0;
			}
				
			// safety
			if (inputLineSplit[5].equals("high")) {
				cleanData[5] = 2;
			} else if (inputLineSplit[5].equals("med")) {
				cleanData[5] = 1;
			} else if (inputLineSplit[5].equals("low")) {
				cleanData[5] = 0;
			}

			// car acceptability
			if (inputLineSplit[6].equals("vgood")) {
				cleanData[6] = 3;
			} else if (inputLineSplit[6].equals("good")) {
				cleanData[6] = 2;
			} else if (inputLineSplit[6].equals("acc")) {
				cleanData[6] = 1;
			} else if (inputLineSplit[6].equals("unacc")) {
				cleanData[6] = 0;
			}
		} else if (dataSetNumber == 2) { // house votes
			if (inputLineSplit[0].equals("democrat")) {
				cleanData[16] = 0;
			} else { // republican
				cleanData[16] = 1;
			}
			for (int i = 1; i < columnCount; i++) {
				if (inputLineSplit[i].equals("?")) {
					cleanData[i - 1] = -1; // set to zero if blank, will fix later
				}
				else if (inputLineSplit[i].equals("y")) {
					cleanData[i - 1] = 1;
				} else { // "n"
					cleanData[i - 1] = 0;
				}
			}
		} else if (dataSetNumber == 3) { // abalone
			if (inputLineSplit[0].equals("M")) { // one-hot data for sex attribute
				cleanData[0] = 1;
				cleanData[1] = 0;
				cleanData[2] = 0;
			} else if (inputLineSplit[0].equals("F")) {
				cleanData[0] = 0;
				cleanData[1] = 1;
				cleanData[2] = 0;
			} else if (inputLineSplit[0].equals("I")) {
				cleanData[0] = 0;
				cleanData[1] = 0;
				cleanData[2] = 1;
			}
			for (int i = 3; i < columnCount; i++) {
				cleanData[i] = Double.parseDouble(inputLineSplit[i - 2]);
			}
		} else if (dataSetNumber == 4) { // computers
			for (int i = 2; i < (columnCount + 2); i++) {
				cleanData[i - 2] = Double.parseDouble(inputLineSplit[i]);
			}
		} else if (dataSetNumber == 5) { // forest fire
			for (int i = 0; i < columnCount; i++) {
				String currentVal = inputLineSplit[i];
				if (i == 2) { // month
					if (currentVal.equals("jan")) {
						cleanData[2] = 0;
					} else if (currentVal.equals("feb")) {
						cleanData[2] = 1;
					} else if (currentVal.equals("mar")) {
						cleanData[2] = 2;
					} else if (currentVal.equals("apr")) {
						cleanData[2] = 3;
					} else if (currentVal.equals("may")) {
						cleanData[2] = 4;
					} else if (currentVal.equals("jun")) {
						cleanData[2] = 5;
					} else if (currentVal.equals("jul")) {
						cleanData[2] = 6;
					} else if (currentVal.equals("aug")) {
						cleanData[2] = 7;
					} else if (currentVal.equals("sep")) {
						cleanData[2] = 8;
					} else if (currentVal.equals("oct")) {
						cleanData[2] = 9;
					} else if (currentVal.equals("nov")) {
						cleanData[2] = 10;
					} else if (currentVal.equals("dec")) {
						cleanData[2] = 11;
					}
				} else if (i == 3) { // day
					if (currentVal.equals("sun")) {
						cleanData[3] = 0;
					} else if (currentVal.equals("mon")) {
						cleanData[3] = 1;
					} else if (currentVal.equals("tue")) {
						cleanData[3] = 2;
					} else if (currentVal.equals("wed")) {
						cleanData[3] = 3;
					} else if (currentVal.equals("thu")) {
						cleanData[3] = 4;
					} else if (currentVal.equals("fri")) {
						cleanData[3] = 5;
					} else if (currentVal.equals("sat")) {
						cleanData[3] = 6;
					}
				} else {
					cleanData[i] = Double.parseDouble(currentVal);
				}
			}
		}

		return cleanData;
	}

	/****
	 * Method: fixMissingValues
	 * Description: fixes missing values in data sets
	****/
	private static void fixMissingValues(int dataSetNumber, double[][] data) {
		int rows = getRowCount(dataSetNumber);
		int columns = getColumnCount(dataSetNumber);
		
		if (dataSetNumber == 0) {
			for (int j = 0; j < columns; j++) {
				double average;
				double sum = 0;
				int count = 0;
				for (int i = 0; i < rows; i++) {
					if (data[i][j] > 0) {
						sum = sum + data[i][j];
						count++;
					}
				} 
				// integer values, so round
				average = Math.round(sum / count);
				for (int i = 0; i < rows; i++) {
					if (data[i][j] == 0) {
						data[i][j] = average;
					}
				}
			}
		} else if (dataSetNumber == 2) {
			for (int j = 0; j < columns; j++) {
				double average;
				double sum = 0;
				int count = 0;
				for (int i = 0; i < rows; i++) {
					if (data[i][j] >= 0) {
						sum = sum + data[i][j];
						count++;
					}
				} 
				// integer values, so round
				average = Math.round(sum / count);
				for (int i = 0; i < rows; i++) {
					if (data[i][j] == -1) {
						data[i][j] = Math.round(average);
					}
				}
			}
		}
	}

    
	/****
	 * Method: standardizeData
	 * Description: standardizes data to equally scale input varialbes - used in regression datasets
	****/
	private static void standardizeData(int dataSetNumber, double[][] data) {
		int rows = getRowCount(dataSetNumber);
		int columns = getColumnCount(dataSetNumber);
		
		for (int j = 0; j < columns; j++) {
			
			// find the mean
			double mean;
			double sum = 0;
			for (int i = 0; i < rows; i++) {
				sum = sum + data[i][j];
			}
			mean = (double) sum / (double) rows;
			
			// find the standard deviation
			double standardDeviation;
			double squaredDifferece  = 0;
			for (int i = 0; i < rows; i++) {
				squaredDifferece = squaredDifferece + ((data[i][j] - mean) * (data[i][j] - mean));
			}
			standardDeviation = Math.sqrt(squaredDifferece / (double) rows);
			
			// calculate the Z score for each entry
			for (int i = 0; i < rows; i++) {
				data[i][j] = (data[i][j] - mean) / standardDeviation;
			}
		}
	}

	/****
	 * Method: splitData
	 * Description: splits data into tuning and testing sets. Tuning gets 20% and testing the other 80%
	****/
	private static double[][] splitData(int dataSetNumber, double[][] data, boolean tuning) {
		int rows = getRowCount(dataSetNumber);
		int yColumn = getColumnCount(dataSetNumber);
		double[][] sortedY = new double[rows][2]; // y value, original row index

		// sort y values
		for (int i = 0; i < rows; i++) {
			sortedY[i][0] = data[i][yColumn - 1];
			sortedY[i][1] = i;
		}
		Arrays.sort(sortedY, (row1, row2) -> Double.compare(row1[0], row2[0]));

		int tuningSize = (rows / 5) + 1; // 20%
		int rowCount;
		if (tuning) { // tuning - split off 20%
			rowCount = tuningSize;
		} else { // remainder of data for testing
			rowCount = rows - tuningSize;
		}

		if (rows % 5 == 0) {
			rowCount++;
		}
		double[][] returnData = new double[rowCount][yColumn];

		int curReturnRow = 0;
		int rowIndex;
		for (int i = 0; i < rows; i++) {
			if (tuning) {
				if ((i % 5) == 0) { // 20% of values
					for (int j = 0; j < yColumn; j++) {
						rowIndex = (int) sortedY[i][1];
						returnData[curReturnRow][j] = data[rowIndex][j];
					}
					curReturnRow++;
				}
			} else {
				if ((i % 5) > 0) { // 80% of values
					for (int j = 0; j < yColumn; j++) {
						rowIndex = (int) sortedY[i][1];
						returnData[curReturnRow][j] = data[rowIndex][j];
					}
					curReturnRow++;
				}
			}
			
		}

		return returnData;
	}


    /****
	 * Method: getRowCount
	 * Description: returns rows in a dataset
	****/
	private static int getRowCount(int dataSetNumber) {
		if (dataSetNumber == 0) { // breast cancer
			return 699;
		} else if (dataSetNumber == 1) { // cars
			return 1728;
		} else if (dataSetNumber == 2) { // house votes
			return 435;
		} else if (dataSetNumber == 3) { // abalone
			return 4177;
		} else if (dataSetNumber == 4) { // computer hardware
			return 209;
		} else if (dataSetNumber == 5) { // forest fires 
			return 517;
		}
		return 0;
	}

	/****
	 * Method: getColumnCount
	 * Description: returns columns in a dataset
	****/
	private static int getColumnCount(int dataSetNumber) {
		if (dataSetNumber == 0) { // breast cancer
			return 10;
		} else if (dataSetNumber == 1) { // cars 
			return 7;
		} else if (dataSetNumber == 2) { // house votes
			return 17;
		} else if (dataSetNumber == 3) { // abalone
			return 11;
		} else if (dataSetNumber == 4) { // computer hardware
			return 8;
		} else if (dataSetNumber == 5) { // forest fires
			return 13;
		}
		return 0;
	}

	/****
	 * Method: getReadFile
	 * Description: returns filename for a dataset
	****/
	private static String getReadFile(int dataSetNumber) {
		if (dataSetNumber == 0) {
			return "data/breast-cancer-wisconsin(1).data";
		} else if (dataSetNumber == 1) {
			return "data/car(1).data";
		} else if (dataSetNumber == 2) {
			return "data/house-votes-84(1).data";
		} else if (dataSetNumber == 3) {
			return "data/abalone(1).data";
		} else if (dataSetNumber == 4) {
			return "data/machine(1).data";
		} else if (dataSetNumber == 5) {
			return "data/forestfires(1).data";
		}
		
		return "";
	}

	/****
	 * Method: getClassification
	 * Description: returns true if classification dataset, false if regression
	****/
	private static boolean getClassification(int dataSetNumber) {
		if (dataSetNumber < 3) {
			return true;
		}
		return false;
	}
}
