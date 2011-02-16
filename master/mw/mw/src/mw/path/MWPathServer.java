package mw.path;

import javax.xml.ws.Endpoint;

public class MWPathServer {

	public static void main(String[] args) throws Exception
	{

		java.net.InetAddress localMachine = java.net.InetAddress.getLocalHost();
		System.out.println("Hostname of local machine: " + localMachine.getCanonicalHostName());
		
		String wsdl = "http://" + localMachine.getCanonicalHostName() + ":42042/MWPathService?wsdl";
		//String wsdl = "http://faui05.informatik.uni-erlangen.de:42042/MWPathService?wsdl";
		
		MWPathServiceInterface pathfinder = new MWMyPathService();
		Endpoint e = Endpoint.publish(wsdl, pathfinder);
		System.out.println("Path Service: " + e.isPublished());
		
		while(true) 
		{
			Thread.sleep(Long.MAX_VALUE);
		}
	}

}
