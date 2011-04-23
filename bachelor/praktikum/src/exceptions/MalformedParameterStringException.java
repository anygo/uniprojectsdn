package exceptions;

public class MalformedParameterStringException extends Exception {
	private static final long serialVersionUID = 1L;
	
	public MalformedParameterStringException() {
		super("Default MalformedFormatStringException");
	}
	
	public MalformedParameterStringException(String msg) {
		super(msg);
	}
}
