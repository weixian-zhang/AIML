

$OutFileName = $PSScriptRoot + "\data\resourcetypes.csv"

$azproviders = az provider list | ConvertFrom-Json

$result = [System.Collections.ArrayList]::new()

foreach ($provider in $azproviders) { 
  foreach ($type in $provider.resourceTypes) 
  { 
    $obj = New-Object -TypeName PSObject -Property @{
      ResourceType = $provider.namespace + "/" + $type.resourceType 
      Namespace = $provider.namespace
      Type =  $type.resourceType 
    }

    $result.Add($obj)
  } 
}

$result | Select-Object ResourceType, Namespace, Type | Export-Csv -Path $OutFileName