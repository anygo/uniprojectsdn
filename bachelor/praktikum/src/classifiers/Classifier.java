package classifiers;

import statistics.DataSet;
import exceptions.*;

public interface Classifier {
	void train(DataSet ds) throws TrainingException;
	void evaluate(DataSet ds) throws EvaluationException;
}
