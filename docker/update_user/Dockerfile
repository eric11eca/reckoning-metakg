FROM eric11eca/meta_kg_5:latest

RUN pip3 install accelerate

ARG DUMMY=unknown
RUN DUMMY=${DUMMY}
RUN mkdir -p /nlp
RUN mkdir -p /nlp/output/
RUN mkdir -p /nlp/tensor/
COPY meta_kg/ /nlp/meta_kg/
COPY peft /nlp/peft/
COPY run_maml.py /nlp/
COPY run_cli.py /nlp/
COPY run_eval.sh /nlp/
COPY run_gpt2.sh /nlp/

ENV WANDB_API_KEY 9edee5b624841e10c88fcf161d8dc54d4efbee29

ENTRYPOINT ["./run_eval.sh"]