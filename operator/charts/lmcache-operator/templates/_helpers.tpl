{{/* Resource suffixes match the published installer names. */}}
{{- define "lmcache-operator.resourceName" -}}
{{- $releaseNameLimit := sub 62 (len .suffix) | int -}}
{{- printf "%s-%s" (.root.Release.Name | trunc $releaseNameLimit | trimSuffix "-") .suffix -}}
{{- end -}}

{{/* Keep the existing Deployment selector stable across installation methods. */}}
{{- define "lmcache-operator.selectorLabels" -}}
control-plane: controller-manager
app.kubernetes.io/name: operator
{{- end -}}

{{- define "lmcache-operator.labels" -}}
{{ include "lmcache-operator.selectorLabels" . }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" | quote }}
{{- end -}}
