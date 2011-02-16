package mw;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.LogManager;

import org.apache.commons.codec.binary.Base64;
import org.apache.commons.httpclient.HttpMethod;
import org.apache.commons.httpclient.HttpStatus;
import org.apache.commons.httpclient.methods.GetMethod;

import com.amazonaws.auth.AWSCredentials;
import com.amazonaws.auth.BasicAWSCredentials;
import com.amazonaws.services.ec2.AmazonEC2Client;
import com.amazonaws.services.ec2.model.DescribeInstancesRequest;
import com.amazonaws.services.ec2.model.DescribeInstancesResult;
import com.amazonaws.services.ec2.model.InstanceStateChange;
import com.amazonaws.services.ec2.model.Placement;
import com.amazonaws.services.ec2.model.RunInstancesRequest;
import com.amazonaws.services.ec2.model.RunInstancesResult;
import com.amazonaws.services.ec2.model.TerminateInstancesRequest;
import com.amazonaws.services.ec2.model.TerminateInstancesResult;
import com.amazonaws.services.elasticloadbalancing.AmazonElasticLoadBalancingClient;
import com.amazonaws.services.elasticloadbalancing.model.CreateLoadBalancerRequest;
import com.amazonaws.services.elasticloadbalancing.model.CreateLoadBalancerResult;
import com.amazonaws.services.elasticloadbalancing.model.DeregisterInstancesFromLoadBalancerRequest;
import com.amazonaws.services.elasticloadbalancing.model.DeregisterInstancesFromLoadBalancerResult;
import com.amazonaws.services.elasticloadbalancing.model.Instance;
import com.amazonaws.services.elasticloadbalancing.model.Listener;
import com.amazonaws.services.elasticloadbalancing.model.RegisterInstancesWithLoadBalancerRequest;
import com.amazonaws.services.elasticloadbalancing.model.RegisterInstancesWithLoadBalancerResult;


public class MWLoadController {
	
	private List<String> instanceIDs;
	private AmazonElasticLoadBalancingClient lb;
	private AmazonEC2Client ec2;
	private final String LB_NAME = "gruppe1-lb";

	public MWLoadController() {
		
		LogManager lm = LogManager.getLogManager();
		lm.reset();
		
		instanceIDs = new ArrayList<String>();
		AWSCredentials credentials = new BasicAWSCredentials("AKIAJ2ML7QE2HKJG6R3A", "JOisbUBy5BNfaU92DEujr9smuxX99vXrZX0n9sFy");
		ec2 = new AmazonEC2Client(credentials);
		ec2.setEndpoint("https://ec2.eu-west-1.amazonaws.com");
		lb = new AmazonElasticLoadBalancingClient(credentials);
		lb.setEndpoint("https://elasticloadbalancing.eu-west-1.amazonaws.com");
		
		List<Listener> listeners = new ArrayList<Listener>();
		listeners.add(new Listener ("tcp", 18081, 18081));
		List<String> zones = new ArrayList <String>();
		zones.add(new String("eu-west-1a"));
		
		CreateLoadBalancerRequest request = new CreateLoadBalancerRequest(LB_NAME, listeners, zones);
		CreateLoadBalancerResult result = lb.createLoadBalancer(request);
		System.out.println(result.getDNSName());

	}
	
	public String runInstance() {

		RunInstancesRequest request = new RunInstancesRequest();
		
		String userData = "group=gruppe1;jar=MWFacebookServer.jar;parameters=http://$I4MW_HOSTNAME:18081/MWFacebookServer?wsdl";
		request.setInstanceType("m1.small");
		request.setKeyName("gruppe1");
		request.setImageId("ami-b36652c7");
		request.setMinCount(1);
		request.setMaxCount(1);
		request.setPlacement(new Placement("eu-west-1a"));
		
		request.withUserData(Base64.encodeBase64String(userData.getBytes()));
		RunInstancesResult result = ec2.runInstances(request);
		
		String id = result.getReservation().getInstances().get(0).getInstanceId();
		
		System.out.print("RunInstancesRequest ");
		while (!getInstanceStatus(id).equalsIgnoreCase("running")) {
			System.out.print(".");
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println();
		
		instanceIDs.add(id);
		System.out.println(id + " is now running");
		
		return id;
	}
	
	public void terminateInstance(String id) {

		TerminateInstancesRequest request = new TerminateInstancesRequest();
		request.withInstanceIds(id);
		TerminateInstancesResult result = ec2.terminateInstances(request);
		List<InstanceStateChange> ls = result.getTerminatingInstances();
		for (InstanceStateChange isc : ls) {
			System.out.println(isc.getInstanceId() + ": " + isc.getPreviousState().getName() + " -> " + isc.getCurrentState().getName());
			instanceIDs.remove(isc.getInstanceId());
		}
	}
	
	public String getInstanceStatus(String idToCheck) {
		
		DescribeInstancesRequest request = new DescribeInstancesRequest();
		ArrayList<String> al = new ArrayList<String>();
		al.add(idToCheck);
		request.setInstanceIds(al);
		
		DescribeInstancesResult result = ec2.describeInstances(request);
		return result.getReservations().get(0).getInstances().get(0).getState().getName();
	}
	
	public String getInstanceDNSName(String id) {
		
		DescribeInstancesRequest request = new DescribeInstancesRequest();
		ArrayList<String> al = new ArrayList<String>();
		al.add(id);
		request.setInstanceIds(al);
		
		DescribeInstancesResult result = ec2.describeInstances(request);
		return result.getReservations().get(0).getInstances().get(0).getPublicDnsName();
	}
	
	public void addInstance(String id) {
		
		String url = "http://" + getInstanceDNSName(id) + ":18081/MWFacebookServer?wsdl";
		
		System.out.print("Waiting for Server to be ready: ");
		while (!checkAvailability(url)) {
			System.out.print(".");
			try {
				Thread.sleep(500);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println(" ready");
		
		RegisterInstancesWithLoadBalancerRequest request = new RegisterInstancesWithLoadBalancerRequest();
		request.setLoadBalancerName(LB_NAME);
		ArrayList<Instance> al = new ArrayList<Instance>();
		al.add(new Instance(id));
		request.setInstances(al);
		
		RegisterInstancesWithLoadBalancerResult result = lb.registerInstancesWithLoadBalancer(request);
		
		System.out.println(result.getInstances().get(0).getInstanceId() + " added to LoadBalancer");
	}
	
	public void removeInstance(String id) {
		
		DeregisterInstancesFromLoadBalancerRequest request = new DeregisterInstancesFromLoadBalancerRequest();
		request.setLoadBalancerName(LB_NAME);
		ArrayList<Instance> al = new ArrayList<Instance>();
		al.add(new Instance(id));
		request.setInstances(al);
		
		DeregisterInstancesFromLoadBalancerResult result = lb.deregisterInstancesFromLoadBalancer(request);
		
		System.out.println(result.getInstances().get(0).getInstanceId() + " removed from LoadBalancer");
	}

	public boolean checkAvailability(String url) {
		
		org.apache.commons.httpclient.HttpClient httpclient = new org.apache.commons.httpclient.HttpClient();
		HttpMethod method = new GetMethod(url);
		
		boolean ret = false;

		try {
			int statusCode = httpclient.executeMethod(method);
			if (statusCode == HttpStatus.SC_OK) {
				ret = true;
			}
		} catch (Exception e) {
			//System.err.println(e.getMessage());
		} finally {
			method.releaseConnection();
		}
		
		return ret;

	}
	
	public String createStandByInstance() {
		
		System.out.print("Initializing Standby Instance ");
		String standBy = runInstance();
		instanceIDs.remove(standBy); 
		
		String standByUrl = "http://" + getInstanceDNSName(standBy) + ":18081/MWFacebookServer?wsdl";
		
		while (!checkAvailability(standByUrl)) {
			System.out.print(".");
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		System.out.println(" done");
		
		return standBy;
	}
	
	public static void main(String args[]) throws Exception {
		
		MWLoadController lc = new MWLoadController();	
		// start "master" instance
		String master = lc.runInstance();
		lc.instanceIDs.remove(master); // soll nie geloescht werden
		
		String masterUrl = "http://" + lc.getInstanceDNSName(master) + ":18081/MWFacebookServer?wsdl";
		
		System.out.print("Initializing Facebook Server: ");
		while (!lc.checkAvailability(masterUrl)) {
			System.out.print(".");
			Thread.sleep(1000);
		}
		System.out.println(" done");
		
		String standByInstance = lc.createStandByInstance();
		
		
		lc.addInstance(master);
		
		System.setProperty("MW_SERVER_URL", masterUrl);
		MWClient client = new MWClient();
		
		while (true) {
			int status = client.status(5);
			System.out.println("Aktuelle Last: " + status);
			if (status > 13) {
				lc.addInstance(standByInstance);
				lc.instanceIDs.add(standByInstance); // add old instance to list
				System.out.println("create new StandBy Instance... "); // and generate new one
				standByInstance = lc.createStandByInstance();
			} else if (status < 5 && lc.instanceIDs.size() > 0) {
				lc.removeInstance(lc.instanceIDs.get(0));
				lc.terminateInstance(lc.instanceIDs.get(0));
			} else {
				Thread.sleep(5000);
			}
		}
	}

	
}
