package exceptions;

import java.io.IOException;

public class InvalidFormatException extends IOException {
	private static final long serialVersionUID = 1L;
	
	public String problem;
	public String fileName;
	public long lineNumber;
	
	public InvalidFormatException(String fileName, long lineNumber, String problem) {
		this.problem = problem;
		this.lineNumber = lineNumber;
		this.fileName = fileName;
	}
	
	public String toString() {
		return "InvalidFormatException in " + fileName + ":" + lineNumber + ": " + problem;
	}
}
