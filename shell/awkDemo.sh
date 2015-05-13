#/bin/sh

# A demo just to show how to use loop (for) and array 
# in awk 

# do initilizations in BEGIN{}
# execute END{} int the end
ls | awk 'BEGIN{
	count=0;
	} 
	{
	for(i = 0; i < NR; ++i)
		name[count++] = $i;
	}
	END{
	for(i = 0; i < NR; ++i)
		print name[i];
	}'

