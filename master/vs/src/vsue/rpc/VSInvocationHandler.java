package vsue.rpc;

import java.io.Serializable;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

import vsue.VSConstants;
import vsue.faults.VSMaybeSemantics;
import vsue.faults.VSSendMultipleTimesSemantics;
import vsue.replica.VSRemoteGroupReference;

@SuppressWarnings("serial")
public class VSInvocationHandler implements InvocationHandler, Serializable {
	private VSRemoteGroupReference groupReference;
	
	public VSInvocationHandler(VSRemoteGroupReference groupReference) {
		this.groupReference = groupReference;		
	}	
	
	@Override
	public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
		if (VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.LAST_OF_MANY
				|| VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.AT_MOST_ONCE) {
			VSSendMultipleTimesSemantics rpcSemantics = new VSSendMultipleTimesSemantics(groupReference);
			return rpcSemantics.invoke(proxy, method, args);
		} else if (VSConstants.RPC_SEMANTICS == VSConstants.RPC_SEMANTICS_ENUM.MAYBE) {
			VSMaybeSemantics rpcSemantics = new VSMaybeSemantics(groupReference);
			return rpcSemantics.invoke(proxy, method, args);
		} else {
			return null;
		}
	}
}
