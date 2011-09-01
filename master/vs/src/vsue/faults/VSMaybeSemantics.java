package vsue.faults;

import java.lang.reflect.Method;
import vsue.communication.VSMessage;
import vsue.communication.VSMessageReply;
import vsue.replica.VSRemoteGroupReference;

public class VSMaybeSemantics extends VSAbstractRPCSemantics {

	public VSMaybeSemantics(VSRemoteGroupReference groupReference) {
		super(groupReference);
	}

	@Override
	public Object invoke(Object proxy, Method method, Object[] args)
			throws Throwable {
		VSMessage message = null;
		VSMessageReply reply = null;
		
		System.out.println("Maybe semantics");

		message = generateMessage(method, args);
		
		reply = sendAndReceive(message);

		if (reply.getReturnValue() instanceof Throwable) {
			throw (Throwable) reply.getReturnValue();
		}

		return reply.getReturnValue();
	}
}
