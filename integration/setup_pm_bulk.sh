set -o xtrace

$GATEWAY_IP=

echo "Setup data router"
docker exec -i datarouter-prov sh -c "curl -X PUT http://dmaap-dr-prov:8080/internal/api/PROV_AUTH_ADDRESSES?val=dmaap-dr-prov\|$GATEWAY_IP -v"
docker exec -i datarouter-prov sh -c "curl -X PUT http://dmaap-dr-prov:8080/internal/api/NODES?val=dmaap-dr-node\|$GATEWAY_IP -v"

echo "Create data router feed for pmmapper"
curl -v -X POST -H "Content-Type:application/vnd.dmaap-dr.feed" -H "X-DMAAP-DR-ON-BEHALF-OF:pmmapper" --data-ascii @./pm-bulk/createFeed.json --post301 --location-trusted -k http://localhost:$DMAAP_DR_PROV_PORT_2
echo "Subscriber pmmaper to data-router feed"
curl -v -X POST -H "Content-Type:application/vnd.dmaap-dr.subscription" -H "X-DMAAP-DR-ON-BEHALF-OF:pmmapper" --data-ascii @./pm-bulk/addSubscriber.json --post301 --location-trusted -k http://localhost:$DMAAP_DR_PROV_PORT_2/subscribe/1