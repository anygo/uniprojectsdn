package exceptions;

public class OutOfVocabularyException extends Exception {
	private static final long serialVersionUID = 1L;
	public OutOfVocabularyException() {
		super();
	}
	public OutOfVocabularyException(String msg) {
		super(msg);
	}
}
