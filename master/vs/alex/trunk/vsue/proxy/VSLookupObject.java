package vsue.proxy;

import java.io.Serializable;

public class VSLookupObject implements Serializable {

	private String m_genericInterfaceClassName;
	
	public VSLookupObject(Class interfaceClass) {
		
		m_genericInterfaceClassName = interfaceClass.getCanonicalName();
		
	}
	
	public Class getInterfaceClass() throws ClassNotFoundException {
		
		return Class.forName(m_genericInterfaceClassName); 

	}
	
}

