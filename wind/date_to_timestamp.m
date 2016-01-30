function [t,h] = date_to_timestamp(d)

year = floor(d/1e6);
month = floor(mod(d,1e6)/1e4);
day = floor(mod(d,1e4)/1e2);
h = mod(d,1e2);

t = datenum(year, month, day, h, 0, 0);

