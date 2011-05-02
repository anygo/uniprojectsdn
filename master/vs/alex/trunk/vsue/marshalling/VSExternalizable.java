package vsue.marshalling;

import java.io.Serializable;
import java.io.IOException;

public interface VSExternalizable extends Serializable {

	public void writeExternal(VSObjectOutputStream out) throws IOException;
	public void readExternal(VSObjectInputStream in) throws IOException;

}

