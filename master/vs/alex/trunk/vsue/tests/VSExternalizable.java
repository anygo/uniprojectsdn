package vsue.tests;

import java.io.IOException;
import java.io.Serializable;

import vsue.marshalling.VSObjectInputStream;
import vsue.marshalling.VSObjectOutputStream;

public interface VSExternalizable extends Serializable {
	public void readExternal(VSObjectInputStream in) throws IOException;
	public void writeExternal(VSObjectOutputStream out) throws IOException;
}
