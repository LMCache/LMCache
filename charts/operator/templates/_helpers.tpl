{{/*
Common labels applied to all resources.
*/}}
{{- define "lmcache-operator.labels" -}}
app.kubernetes.io/name: lmcache-operator
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/component: controller-manager
{{- with .Values.commonLabels }}
{{ toYaml . }}
{{- end }}
{{- end -}}

{{/*
Selector labels for the operator pod.
*/}}
{{- define "lmcache-operator.selectorLabels" -}}
control-plane: controller-manager
app.kubernetes.io/name: lmcache-operator
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Full image reference (repository:tag).
*/}}
{{- define "lmcache-operator.image" -}}
{{- printf "%s:%s" .Values.image.repository .Values.image.tag -}}
{{- end -}}
