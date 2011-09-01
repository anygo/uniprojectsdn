package vsue.replica;

import java.io.Serializable;

import vsue.rpc.VSRemoteReference;

@SuppressWarnings("serial")
public class VSRemoteGroupReference implements Serializable {
	private VSRemoteReference[] references;

	public VSRemoteGroupReference()
	{
		
	}
	
	public void setReferences(VSRemoteReference[] references) {
		this.references = references;
	}

	public VSRemoteReference[] getReferences() {
		return references;
	}
	
}
