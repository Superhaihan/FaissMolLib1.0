How to Run？

Step one, run the graph_compute.py file, which will generate all the embedding.npy files, id2name.json files, and name2id.json files in the library.

Step two, run the search_deploy.py file to add all the vectors in the library to the faiss training and deploy it as a query service.

Step three, run the send_request.py and app.py file, send a query request to the query service, and return the query result.