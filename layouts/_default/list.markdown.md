# {{ .Title }}

Source: {{ .Permalink }}

{{ with .Description }}{{ . }}

{{ end -}}
{{ with .RawContent }}{{ . }}

{{ end -}}
{{ range .Pages.ByDate.Reverse -}}
- [{{ .Title }}]({{ .Permalink }}){{ with .Date }} — {{ .Format "2006-01-02" }}{{ end }}
{{ end -}}
