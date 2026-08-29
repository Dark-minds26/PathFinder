PROJECTS={
 "python_basics":("Python Foundations CLI",["python_basics"],4,"Build a small CLI that reads data, validates input, and produces a useful report."),
 "docker_basics":("Containerize an ML API",["docker_basics","rest_apis"],8,"Package a small API into a Docker image with environment-based configuration and a health check."),
 "machine_learning":("End-to-end ML Predictor",["machine_learning","python_advanced"],10,"Train, evaluate and expose a small prediction service with reproducible preprocessing."),
 "llm_applications":("LLM Q&A Assistant",["llm_applications"],10,"Build a grounded question-answering app with prompt templates, validation and evaluation."),
 "rag_systems":("RAG Knowledge Assistant",["rag_systems","llm_applications"],12,"Ingest documents, create embeddings, retrieve relevant chunks and evaluate grounded answers."),
 "model_serving":("Production Model Service",["model_serving","mlops_basics"],12,"Serve a trained model behind an API with health checks, containerization and basic monitoring."),
}
def project_for(skill_id):
    p=PROJECTS.get(skill_id)
    if not p:return {"project_id":f"project_{skill_id}","title":f"{skill_id.replace('_',' ').title()} Practice Project","skills":[skill_id],"estimated_hours":6,"description":f"Build a small portfolio project demonstrating {skill_id.replace('_',' ')}."}
    title,skills,hours,desc=p; return {"project_id":"project_"+skill_id,"title":title,"skills":skills,"estimated_hours":hours,"description":desc}
