FROM eric11eca/meta_kg_user:latest


ARG DUMMY=unknown
RUN DUMMY=${DUMMY}
RUN mkdir -p /nlp
RUN mkdir -p /nlp/output/
RUN mkdir -p /nlp/data/
COPY meta_kg/ /nlp/meta_kg/
COPY scripts/ /nlp/scripts/
COPY config /nlp/config
COPY run_maml.py /nlp/
COPY run_cli.py /nlp/

ENV WANDB_API_KEY 9edee5b624841e10c88fcf161d8dc54d4efbee29

CMD ["/bin/bash"]