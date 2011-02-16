package mw.path;

public class MWNoPathException extends Exception {
	private static final long serialVersionUID = 6583941272052047359L;
	
	@Override
	public String getMessage()
	{
		return "Es konnte kein Pfad gefunden werden (eventuell ist MAX_ITER erreicht)";
	}

}
